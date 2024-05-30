from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
from copy import deepcopy

from util.data_util import check_dataset
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.factory import get_imdb
from dataset.roidb import RoiDataset, detection_collate, Custom_yolo_dataset
from yolov2_tiny_2 import Yolov2
# from yolov2_tiny_LightNorm import Yolov2
from torch import optim
from torch.optim import lr_scheduler
from util.network import adjust_learning_rate
from tensorboardX import SummaryWriter
from config import config as cfg
from Test_with_train import test_for_train
from weight_update import *
import cv2
from PIL import Image
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Yolo v2')
    parser.add_argument('--max_epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=160, type=int)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=1, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                        default=8, type=int)
    parser.add_argument('--data_limit', dest='data_limit',
                        default=0, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        default='voc0712trainval', type=str)
    parser.add_argument('--data', type=str,
                        default=None, help='Give the path of custom data yaml file' )
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
    parser.add_argument('--checkpoint_epoch', dest='checkpoint_epoch',
                        default=160, type=int)
    parser.add_argument('--exp_name', dest='exp_name',
                        default='default', type=str)

    args = parser.parse_args()
    return args

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

def util(check_point):
    dum = []
    for i, (k,v) in enumerate(check_point.items()):
        if k == 'conv9.0.weight':    #con9: torch.Size([40, 1024, 1, 1]), bias9: torch.Size([40])
            v = torch.rand((40, 1024, 1, 1))
            v /= 1000
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
def train():
    
    # define the hyper parameters first
    args = parse_args()
    args.lr = cfg.lr
    # args.decay_lrs = cfg.decay_lrs
    args.weight_decay = cfg.weight_decay
    args.momentum = cfg.momentum
    args.batch_size = args.batch_size
    # args.data_limit = 80
    # args.pretrained_model = os.path.join('data', 'pretrained', 'darknet19_448.weights')
    # args.pretrained_model = os.path.join('yolov2-pytorch','data', 'pretrained', 'yolov2-tiny-voc.pth') #cHANGE
    args.pretrained_model = os.path.join('data', 'pretrained', 'yolov2-tiny-voc.pth')
    print('Called with args:')
    print(args)

    if args.dataset == 'custom':
        data_dict = check_dataset(args.data)
        train_path, val_path, val_dir = data_dict['train'], data_dict['val'], data_dict['val_dir']
        nc = int(data_dict['nc'])  # number of classes
        names = data_dict['names']  # class names
        assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {args.data}'  # check
        train_dataset = Custom_yolo_dataset(train_path)
        # args.customData = True
        # args.withTrain = True
        args.val_dir = val_dir
        
    else:    
        args.imdb_name, args.imdbval_name = get_dataset_names(args.dataset)
        # load dataset
        print('loading dataset....')
        train_dataset = get_dataset(args.imdb_name)
        val_path, nc = None, None       #not needed for voc
    
    output_dir = os.path.join(os.getcwd(),args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    if not args.data_limit==0:
        train_dataset = torch.utils.data.Subset(train_dataset, range(0, args.data_limit))
    print('dataset loaded.')

    print('Training Dataset: {}'.format(len(train_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,    #args.batch_size
                                  shuffle=True, num_workers=args.num_workers,      # args.num_workers
                                  collate_fn=detection_collate, drop_last=True, pin_memory=True)
    
    
    iters_per_epoch = int(len(train_dataset) / args.batch_size)
    print("Training Iterations: " + str(iters_per_epoch)+"\n")
        
    # Read pre-trained weights
    if args.resume:
        pre_trained_checkpoint = torch.load(args.pretrained_model,map_location='cpu')
        modified = util(pre_trained_checkpoint['model']) 

    # initialize the model
    print('initialize the model')
    try: nc
    except: nc = None

    EpochPerRound = 10
    num_of_clients = 4
    data_mode = "homogenous" # "unique" "hetrogenous"
    
    models, optimizers = [], []
    min_losses, max_maps, best_map_scores, best_map_epochs, best_map_losses, map = [], [], [], [], [], []
    client_best_weights           = dict()
    client_best_weights["models"] = dict()
    client_best_weights["epoch" ] = dict()
    client_best_weights["loss"  ] = dict()
    client_best_weights["map"   ] = dict()


    # Initialize Aggregate Model        
    if nc is not None:
        aggr_model = Yolov2(classes=names)
    else:
        aggr_model = Yolov2()
    if args.use_cuda:   aggr_model.cuda()
    
        
    # Initialize Client Models
    for _c in range(num_of_clients):
        if nc is not None:
            models.append(Yolov2(classes=names))
        else:
            models.append(Yolov2())
    
    for i, _m in enumerate(models):
        print(f"Initializing Client Model {i}")
        if args.resume: _m.load_state_dict(modified['model'])
        optimizers.append(optim.SGD(_m.parameters(), 
                                    lr=args.lr, 
                                    momentum=args.momentum, 
                                    weight_decay=args.weight_decay ))
        schedulers = (lr_scheduler.MultiStepLR(optimizers[i], milestones=[8,70,100,130], gamma=0.1))
        if args.use_cuda:   _m.cuda()
        if args.mGPUs:      _m = nn.DataParallel(_m)
        if True:            _m.train()
        
        min_losses.append(      100) 
        max_maps.append(          0) 
        best_map_scores.append(  -1) 
        best_map_epochs.append(  -1) 
        best_map_losses.append(  -1) 
        map.append(               0)
            

        # # Check and save the best mAP
        # save_name_temp = os.path.join(output_dir, 'temp')
        # if args.dataset == 'custom':
        #     map, _ = test_for_train(save_name_temp, model, args, val_data=val_path, _num_classes = nc)
        # else:
        #     map = test_for_train(save_name_temp, model, args)
        # print(f'\t-->>Initial mAP - Before starting training={round((map*100),2)}')
        
    # Start training
    for epoch in range(args.start_epoch, args.max_epochs+1):
        # variables to store key params in an epoch
        loss_temp = np.zeros(len(models))
        box_losses, iou_losses, class_losses = [], [], []
        # Data Iterator
        train_data_iter = iter(train_dataloader)
    
        # Run for all batches OR iterations per Epoch
        for step in tqdm(range(iters_per_epoch), desc=f'Epoch {epoch}', total=iters_per_epoch):

            # Randomly select a scale from the specified range
            if cfg.multi_scale and (step + 1) % cfg.scale_step == 0:
                scale_index = np.random.randint(*cfg.scale_range)
                cfg.input_size = cfg.input_sizes[scale_index]

            # Get the next batch of training data
            im_data, boxes, gt_classes, num_obj = next(train_data_iter)

            # Move the data tensors to the GPU
            if args.use_cuda:
                im_data, boxes, gt_classes, num_obj = \
                    im_data.cuda(), boxes.cuda(), gt_classes.cuda(), num_obj.cuda()

            # Convert the input data tensor to a PyTorch Variable
            im_data_variable = Variable(im_data)
            
            if data_mode=="same":
                for i, (_model, _opt, _loss) in enumerate(zip(models, optimizers)):
                    # Forward Pass
                    box_loss, iou_loss, class_loss = _model(im_data_variable, boxes, gt_classes, num_obj, training=True)
                    
                    # Compute the total loss
                    loss = box_loss.mean() + iou_loss.mean() + class_loss.mean()
                    
                    # Clear gradients
                    _opt.zero_grad()
                    
                    # Compute gradients
                    loss.backward()

                    # Update weights
                    _opt.step()
                
                    # Update loss values
                    box_losses.append(box_loss)
                    iou_losses.append(iou_loss)
                    class_losses.append(class_loss)
                    loss_temp[i] += loss.item()

            elif data_mode == "homogenous":
    
                # Client Number to use
                i = step%num_of_clients
                _model, _opt = models[i], optimizers[i]
            
                # Forward Pass
                box_loss, iou_loss, class_loss = _model(im_data_variable, boxes, gt_classes, num_obj, training=True)
                
                # Compute the total loss
                loss = box_loss.mean() + iou_loss.mean() + class_loss.mean()
                
                # Clear gradients
                _opt.zero_grad()
                
                # Compute gradients
                loss.backward()

                # Update weights
                _opt.step()
            
                # Update loss values
                box_losses.append(box_loss)
                iou_losses.append(iou_loss)
                class_losses.append(class_loss)
                loss_temp[i] += loss.item()
                    
        # Show loss after epoch
        loss_temp /= iters_per_epoch
        
        for i, (_model, _opt) in enumerate(models, optimizers):
            iou_loss_v      = iou_losses[   i].mean().item()
            box_loss_v      = box_losses[   i].mean().item()
            class_loss_v    = class_losses[ i].mean().item()

            print(f"[epoch %2d][step %4d/%4d] loss: %.4f, lr: %.2e, tiou_loss: %.4f, box_loss: %.4f, class_loss: %.4f" \
                    % (epoch, step+1, iters_per_epoch, loss_temp, _opt.param_groups[0]['lr'], iou_loss_v, box_loss_v, class_loss_v), end =' ')

            # if epoch % args.save_interval == 0:
            #     save_name = os.path.join(output_dir, f'yolov2_client{i}_epoch_{epoch}.pth')
            #         torch.save({
            #                     'model' : _model.module.state_dict() if args.mGPUs else _model.state_dict(),
            #                     'epoch' : epoch,
            #                     'lr'    : _opt.param_groups[0]['lr'],
            #                     'loss'  : loss.item()
            #                     }, 
            #                    save_name)
                
            # # Check minimum loss and save weights for minimum loss
            # if loss.item() < min_losses[i]:
            #     min_loss[i] = loss.item()
            #     print(f'\n\t-->>Saving lower loss weights at Epoch {epoch}, with loss={round(loss.item(),2)}')
            #     save_name = os.path.join(output_dir, 'yolov2_client{}_least_loss.pth'.format(i))
            #     torch.save({
            #         'model': models[i].state_dict(),
            #         'epoch': epoch,
            #         'lr': optimizers[i].param_groups[0]['lr'],
            #         'loss': loss.item()
            #         }, save_name)
            
            # Check and save the best mAP
            save_name_temp = os.path.join(output_dir, 'temp')
            
            # Original
            map[i], _ = test_for_train(save_name_temp, _model, args, val_path, nc)
            
            if map[i] > max_maps[i]:
                max_maps[i] = map[i]
                best_map_scores[i] = round((map*100),2)
                best_map_epochs[i] = epoch
                best_map_losses[i]  = round(loss.item(),2)
                
                client_best_weights["models"][i] = models[i].state_dict()
                client_best_weights["epoch" ][i] = epoch
                client_best_weights["loss"  ][i] = loss_temp[i].item()
                client_best_weights["map"   ][i] = map
                # save_name = os.path.join(output_dir, 'yolov2_client{}_best_map.pth'.format(i))
                # print(f'\n\t--------------------->>Saving best weights at Epoch {epoch}, with mAP={round((map*100),2)}% and loss={round(loss.item(),2)}\n')
                # torch.save({
                #     'model': models[i].state_dict(),
                #     'epoch': epoch,
                #     'lr': optimizers[i].param_groups[0]['lr'],
                #     'loss': loss.item(),
                #     'map': map
                #     }, save_name)
        
            print(f'\n\t---------------------Best mAP for client{i} was at Epoch {best_map_epochs[i]}, with mAP={best_map_scores[i]}% and loss={best_map_losses[i]}\n')
            
        # # step learning rate
        for i, _sch in enumerate(schedulers):
            _sch.step()        
        # weight aggregation
        if epoch % EpochPerRound == 0 or epoch == args.max_epochs:
            print('Aggregate weights and load best')
            
            # Aggregate model
            for name, param in aggr_model.named_parameters():
                print(f"Aggregating {name}")
                sum = 0
                data= []
                for _m in models:
                    _m_state = _m.state_dict()
                    data.append(_m_state[name].detach().cpu().numpy())
                aggr_data = np.sum(np.array(data), axis=0) 
                param.data = torch.from_numpy(aggr_data).cuda()
                
            # Validate aggregated model
            # agg_map = []
            agg_map, _ = test_for_train(save_name_temp, _model, args, val_path, nc)
            
            # Identify best model
            all_maps = deepcopy(best_map_scores)
            all_maps.append(agg_map)
            best_map = max(all_maps)
            _index   = all_maps.index()
            if _index==num_of_clients: best_model = aggr_model
            else: best_model = models[_index]
            
            # Update client models
            for _m in models:
                _m.load_state_dict(best_model.state_dict())
                
                    
                    
if __name__ == '__main__':
    train()