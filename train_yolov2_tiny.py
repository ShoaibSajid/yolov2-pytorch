from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os, traceback
import numpy as np
import argparse
import time
import torch
import torch.nn as nn

from util.data_util import check_dataset
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
# import torchdata.datapipes.iter as pipes
from dataset.factory import get_imdb
from dataset.roidb import RoiDataset, detection_collate, Custom_yolo_dataset
from yolov2_tiny_2 import Yolov2
# from yolov2_tiny_LightNorm import Yolov2
from torch import optim
from torch.optim import lr_scheduler
from util.network import adjust_learning_rate
# from tensorboardX import SummaryWriter
from config import config as cfg
from Test_with_train import test_for_train
from weight_update import *
import cv2
from PIL import Image
from collections import OrderedDict
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)


torch.manual_seed(0)
np.random.seed(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

@contextlib.contextmanager
def num_torch_thread(n_thread: int):
    n_thread_original = torch.get_num_threads()
    torch.set_num_threads(n_thread)
    yield
    torch.set_num_threads(n_thread_original)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Yolo v2')
    parser.add_argument('--max_epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=100, type=int)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=1, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                        default=64, type=int)
    parser.add_argument('--data_limit', dest='data_limit',
                        default=0, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        default='voc0712trainval', type=str)
    parser.add_argument('--data', type=str, dest='data',
                        default=None, help='Give the path of custom data .yaml file' )
    parser.add_argument('--nw', dest='num_workers',
                        help='number of workers to load training data',
                        default=8, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                        default='output', type=str)
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        default=False, type=bool)
    parser.add_argument('--display_interval', dest='display_interval',
                        default=20, type=int)
    parser.add_argument('--mGPUs', dest='mGPUs',
                        default=False, type=bool)
    parser.add_argument('--save_interval', dest='save_interval',
                        default=10, type=int)
    parser.add_argument('--cuda', dest='use_cuda',
                        default=True, type=bool)
    parser.add_argument('--resume', dest='resume',
                        default=False, type=bool)
    parser.add_argument('--weights', default='', dest='weights',
                        help='provide the path of weight file (.pth) if resume')
    parser.add_argument('--checkpoint_epoch', dest='checkpoint_epoch',
                        default=100, type=int)
    parser.add_argument('--exp_name', dest='exp_name',
                        default='default', type=str)
    parser.add_argument('--device', default=0, dest='device',
                        help='Choose a gpu device 0, 1, 2 etc.')
    parser.add_argument('--savePath', default='results',
                        help='')
    parser.add_argument('--imgSize', default='1280,720',
                        help='image size w,h') 
    parser.add_argument('--cleaning', dest='cleaning', 
                        default=False, type=bool,
                        help='Set true to remove small objects')
    parser.add_argument('--pix_th', dest='pix_th', 
                        default=12, type=int,
                        help='Pixel Threshold value') 
    parser.add_argument('--asp_th', dest='asp_th', 
                        default=1.5, type=float,
                        help='Aspect Ratio threshold')
    args = parser.parse_args()
    return args

def util(check_point):
    dum = []
    for i, (k,v) in enumerate(check_point.items()):
        if k == 'conv9.0.weight':    #con9: torch.Size([40, 1024, 1, 1]), bias9: torch.Size([40])
            v = torch.rand((40, 1024, 1, 1))
            v /= 100000
            append = (k,v)
            dum.append(append)
        elif k == 'conv9.0.bias':
            v = torch.rand(40)
            v /= 1000
            append = (k,v)
            dum.append(append)
        else:
            append = (k,v)
            dum.append(append)
            # print(v)            
    modified_check_point = {"model": OrderedDict(dum)}
    return modified_check_point

def get_dataset(datasetnames):
    names = datasetnames.split('+')
    dataset = RoiDataset(get_imdb(names[0]))
    print('load dataset {}'.format(names[0]))
    for name in names[1:]:
        tmp = RoiDataset(get_imdb(name))
        dataset += tmp
        print('load and add dataset {}'.format(name))
    return dataset

def drawBox(label:np.array, img:np.ndarray):
    # for i in range(label.shape[0]):
    h, w, _ = img.shape
    box = [label[0], label[1], label[2], label[3]]
    img = cv2.rectangle(img,(int(box[0]*w), int(box[1]*h)), (int(box[2]*w), int(box[3]*h)), (0,0,255), 1)
    return img

def showImg(img, labels, std=None, mean=None):
    # Convert the tensor to a numpy array
    _image = img
    image_np = _image.numpy().transpose((1, 2, 0))
    # image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)*255
    _img = Image.fromarray(image_np.astype('uint8'), 'RGB')
    _img = np.array(_img)
    _img = cv2.cvtColor(_img, cv2.COLOR_RGB2BGR)
    for i in range(labels.shape[0]):
        label = labels[i].numpy()
        _img = drawBox(label, _img)
    cv2.imshow('', _img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def nan_hook(self, inp, output):
            if not isinstance(output, tuple):
                outputs = [output]
            else:
                outputs = output

            for i, out in enumerate(outputs):
                nan_mask = torch.isnan(out)
                if nan_mask.any():
                    print("In", self.__class__.__name__)
                    raise RuntimeError(f"Found NAN in output {i} at indices: ", 
                                       nan_mask.nonzero(), 
                                       "where:", 
                                       out[nan_mask.nonzero()[:, 0].unique(sorted=True)] if nan_mask.nonzero().size()[1]>0 else out)

def train(args, device=None):
    
    if device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = f'{device}'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.device}'   
    try: args.lr
    except: args.lr=None
    if not args.lr:
        args.lr = cfg.lr
    # args.decay_lrs = cfg.decay_lrs
    args.weight_decay = cfg.weight_decay
    args.momentum = cfg.momentum
    # args.batch_size = args.batch_size
    # args.data_limit = 80
    # args.pretrained_model = os.path.join('data', 'pretrained', 'darknet19_448.weights')
    # args.pretrained_model = os.path.join('data', 'pretrained', 'yolov2-tiny-voc.pth') #cHANGE
    # args.pretrained_model = os.path.join('data', 'pretrained', 'yolov2_least_loss_waymo.pth') #cHANGE

    print('Called with args:')
    print(args)
    
    if args.dataset == 'custom':
        args.scaleCrop = True
        data_dict = check_dataset(args.data)
        train_path, val_path, val_dir = data_dict['train'], data_dict['val'], data_dict['val_dir']
        nc = int(data_dict['nc'])  # number of classes
        names = data_dict['names']  # class names
        assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {args.data}'  # check
        print(f'loading training data from {Fore.GREEN}{Style.BRIGHT}{train_path}....')
        train_dataset = Custom_yolo_dataset(train_path,
                                            cleaning=args.cleaning,
                                            pix_th=args.pix_th,
                                            asp_th=args.asp_th,
                                            scale_Crop=args.scaleCrop)
        # train_dataset = pipes.InMemoryCacheHolder(_train_dataset,size=2048).sharding_filter()
        args.val_dir = val_dir
        _nc = nc
    else:
        args.imdb_name, args.imdbval_name = get_dataset_names(args.dataset)
        _nc = 20
        # load dataset
        print('loading dataset....')
        train_dataset = get_dataset(args.imdb_name)
    
    _output_dir = os.path.join(os.getcwd(), args.output_dir)
    if not os.path.exists(_output_dir):
        os.makedirs(_output_dir)

    
    if not args.data_limit==0:
        train_dataset = torch.utils.data.Subset(train_dataset, range(0, args.data_limit))
    print(f'{Style.BRIGHT}dataset loaded....')

    print(f'{Fore.GREEN}{Style.BRIGHT}Training Dataset: {len(train_dataset)}')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,    #args.batch_size
                                  shuffle=True, num_workers=args.num_workers,      # args.num_workers
                                  collate_fn=detection_collate, drop_last=True, pin_memory=True)

    # initialize the model
    print(f'{Style.BRIGHT}initialize the model....')
    tic = time.time()
    try:
        nc, args.fedml
    except:
        nc = None
        args.fedml = False

    if not args.fedml:
        if nc is not None:
            model = Yolov2(classes=names)
        else:
            model = Yolov2()    
        
        # for submodule in model.modules():
        #     submodule.register_forward_hook(nan_hook)

        if args.resume:     # False
            print(f'Loading the pre-trained checkpoint from {Fore.GREEN}{Style.BRIGHT}{args.weights}....')
            # pre_trained_checkpoint = torch.load(args.pretrained_model,map_location='cpu') #---CHANGE
            pre_trained_checkpoint = torch.load(args.weights,map_location='cpu') #---CHANGE
            # model.load_state_dict(pre_trained_checkpoint['model'])
            _model = pre_trained_checkpoint['model']
            if _model['conv9.0.weight'].shape[0] != (5+_nc)*5:
                print(f'{Style.BRIGHT}Last layer of pretrain checkpoint is different')
                print(f'{Style.BRIGHT}Changing the last layer of {Fore.MAGENTA}{Style.BRIGHT}{args.weights}...')
                pre_trained_checkpoint = util(_model)    # con9: torch.Size([40, 1024, 1, 1]), bias9: torch.Size([40])
            # check_point={k:v if v.size()==model[k].size()  else  model[k] for k,v in zip(enumerate(model.items()), enumerate(pre_trained_checkpoint["model"].items()))}
            
            try: model.load_state_dict(pre_trained_checkpoint['model'])
            except Exception as e:
                traceback.print_exc()
                print(f"Error: {e}")    
    
    try:args.model
    except:args.model=None
    if args.model is not None:
        model = args.model
    
    toc = time.time()
    print('model loaded: cost time {:.2f}s'.format(toc-tic))
    
    try:
        args.fedml
    except:
        args.fedml = False

    # initialize the optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if not args.fedml:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,60,120,150], gamma=0.1)
    try:scheduler
    except:scheduler=False
    if args.use_cuda:
        model.cuda()

    if args.mGPUs:
        model = nn.DataParallel(model)

    # set the model mode to train because we have some layer whose behaviors are different when in training and testing.
    # such as Batch Normalization Layer.
    best_model = model
    model.train()

    iters_per_epoch = int(len(train_dataset) / args.batch_size)
    print("Training Iterations: " + str(iters_per_epoch)+"\n")

    min_loss = 100
    max_map = 0
    best_map_score = -1     
    best_map_epoch = -1     
    best_map_loss  = -1 

    # # Check and save the best mAP
    save_name_temp = os.path.join(_output_dir, 'temp')
    # if args.dataset == 'custom':
    #     map, _ = test_for_train(_output_dir, model, args, val_data=val_path, classes = names)
    # else:
    #     map, _ = test_for_train(_output_dir, model, args)
    # print(f'\t-->>Initial mAP - Before starting training={round((map*100),2)}')
    
    # Start training
    for epoch in range(args.start_epoch, args.max_epochs+1):
        loss_temp = 0
        tic = time.time()
        train_data_iter = iter(train_dataloader)
        print()
        # optimizer.zero_grad()
        for step in tqdm(range(iters_per_epoch), desc=f'Epoch {epoch}', total=iters_per_epoch):

            # Randomly select a scale from the specified range
            if cfg.multi_scale and (step + 1) % cfg.scale_step == 0:
                scale_index = np.random.randint(*cfg.scale_range)
                cfg.input_size = cfg.input_sizes[scale_index]

            # Get the next batch of training data
            # print('Loading first batch of images')
            im_data, boxes, gt_classes, num_obj, im_info = next(train_data_iter)     #boxes=[b, 4,4] ([x1,x2,y1,y2]) padded with zeros
            # for i in range(im_data.shape[0]):
            #     showImg(im_data[i], boxes[i])

            # Move the data tensors to the GPU
            if args.use_cuda:
                im_data = im_data.cuda()
                boxes = boxes.cuda()
                gt_classes = gt_classes.cuda()
                num_obj = num_obj.cuda()

            # Convert the input data tensor to a PyTorch Variable
            im_data_variable = Variable(im_data)

            # Compute the losses
            box_loss, iou_loss, class_loss = model(im_data_variable, boxes, gt_classes, num_obj, training=True, im_info=im_info)

            # Compute the total loss
            loss = box_loss.mean() + iou_loss.mean() + class_loss.mean() 
            
            # Clear gradients
            optimizer.zero_grad()
            # Compute gradients
            # loss.retain_grad()
            # loss.backward(retain_graph=True)
            loss.backward()

            # Gradient clipping to prevent nans
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10., norm_type=2)

            optimizer.step()

            loss_temp += loss.item()
        if scheduler:
            scheduler.step()
        # Show loss after epoch
        toc = time.time()
        loss_temp /= iters_per_epoch

        iou_loss_v = iou_loss.mean().item()
        box_loss_v = box_loss.mean().item()
        class_loss_v = class_loss.mean().item()

        print(f"[epoch %2d][step %4d/%4d] loss: %.4f, lr: %.2e, time cost %.1fs, tiou_loss: %.4f, box_loss: %.4f, class_loss: %.4f" \
                % (epoch, step+1, iters_per_epoch, loss_temp, optimizer.param_groups[0]['lr'], toc - tic, iou_loss_v, box_loss_v, class_loss_v), end =' ')

        loss_temp = 0
        tic = time.time()

        if epoch % args.save_interval == 0:
            save_name = os.path.join(_output_dir, 'yolov2_epoch_{}.pth'.format(epoch))
            torch.save({
                'model': model.module.state_dict() if args.mGPUs else model.state_dict(),
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr'],
                'loss': loss.item()
                }, save_name)
            
        # Check minimum loss and save weights for minimum loss
        if loss.item() < min_loss:
            min_loss = loss.item()
            print(f'\n\t-->>Saving lower loss weights at Epoch {epoch}, with loss={round(loss.item(),2)}')
            save_name = os.path.join(_output_dir, 'yolov2_least_loss.pth')
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr'],
                'loss': loss.item()
                }, save_name)
        
        # Check and save the best mAP
        save_name_temp = os.path.join(_output_dir, 'temp')
        
        if args.dataset == 'custom':
            with num_torch_thread(1):
                map, _ = test_for_train(_output_dir, model, args, val_path, names)
        else:
            map, _ = test_for_train(_output_dir, model, args)
        if map > max_map:
            best_model = model
            max_map = map
            best_map_score = round((map*100),2)
            best_map_epoch = epoch
            best_map_loss  = round(loss.item(),2)
            save_name_best = os.path.join(_output_dir, 'yolov2_best_map.pth')
            print(f'\n\t--------------------->>Saving best weights at Epoch {epoch}, with mAP={round((map*100),2)}% and loss={round(loss.item(),2)}\n')
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr'],
                'loss': loss.item(),
                'map': map
                }, save_name_best)
                
    if not args.fedml:
        print(f'\n\t---------------------{Style.BRIGHT}Best mAP was at Epoch {best_map_epoch}, with mAP={best_map_score}% and loss={best_map_loss}\n')
        print(f'{Style.BRIGHT}Validating after Training...')
        print(f'Loading best weights from {Style.BRIGHT}{Fore.GREEN}{save_name_best}')
        checkpoint = torch.load(save_name_best,map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        map, _ = test_for_train(_output_dir, model, args, val_path, names, True)
    else:
        return best_model, max_map    



if __name__ == '__main__':
    # define the hyper parameters first
    args = parse_args()
    train(args)