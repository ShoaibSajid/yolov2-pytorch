from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import numpy as np
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
# from yolov2_tiny_2 import Yolov2
# from yolov2_tiny_LightNorm import Yolov2
from torch import optim
from torch.optim import lr_scheduler
# from util.network import adjust_learning_rate
# from tensorboardX import SummaryWriter
from config import config as cfg
from Test_with_train import test_for_train
from weight_update import *
import copy
# import cv2
# from PIL import Image
# from collections import OrderedDict
import colorama
from colorama import Fore, Back, Style
colorama.init(autoreset=True)


torch.manual_seed(0)
np.random.seed(0)

@contextlib.contextmanager
def num_torch_thread(n_thread: int):
    n_thread_original = torch.get_num_threads()
    torch.set_num_threads(n_thread)
    yield
    torch.set_num_threads(n_thread_original)
    
def get_dataset(datasetnames):
    names = datasetnames.split('+')
    dataset = RoiDataset(get_imdb(names[0]))
    print('load dataset {}'.format(names[0]))
    for name in names[1:]:
        tmp = RoiDataset(get_imdb(name))
        dataset += tmp
        print('load and add dataset {}'.format(name))
    return dataset
    
class Trainer:
    def __init__(self, shared_memory):
        super().__init__()
    
        self.shared_memory =  shared_memory
    
    def set_args(self, args, clientId):
        # set up the necessary args for client training
        args.data       = args.client_data[clientId]
        args.output_dir = f'{args.output_dir}/client_{clientId}'
        args.model      = self.shared_memory['aggr_model'].to('cpu')
        args.lr         = cfg.lr
        
        args.weight_decay = cfg.weight_decay
        args.momentum     = cfg.momentum
        return args
    
    def loadData(self, args):
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
        else:
            args.imdb_name, args.imdbval_name = get_dataset_names(args.dataset)
            _nc = 20
            # load dataset
            print('loading dataset....')
            train_dataset = get_dataset(args.imdb_name)
            
            val_path, val_dir, names, nc = None
            
        return train_dataset, val_path, val_dir, names, nc
    
    def train_client(self, args, clientId, device, lock):
        
        # os.environ["CUDA_VISIBLE_DEVICES"] = f'{device}'
        
        args = self.set_args(args, clientId)
        
        print('Called with args:')
        print(args)
        
        #load dataset
        train_dataset, val_path, val_dir, names, nc = self.loadData(args)
        
        args.val_dir = val_dir
        _nc = nc
        
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
        model = args.model
        
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5,20], gamma=0.1)
        lr_sch    = [5,10,20]
        ind       = 0

        if args.use_cuda:
            model.to(device)
        
        if args.mGPUs:
            model = nn.DataParallel(model)
        model.train()     
            
        iters_per_epoch = int(len(train_dataset) / args.batch_size)
        print("Training Iterations: " + str(iters_per_epoch)+"\n")

        # min_loss = 100
        max_map = 0
        # best_map_score = -1     
        # best_map_epoch = -1     
        # best_map_loss  = -1
        
        _round = 0
        for epoch in range(args.start_epoch, (args.epochPerRound * args.max_rounds)+1):   
                
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
                    # if not next(model.parameters()).is_cuda:
                    #     model.to(device)
                    # model.train()
                    im_data = im_data.to(device)
                    boxes = boxes.to(device)
                    gt_classes = gt_classes.to(device)
                    num_obj = num_obj.to(device)

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
            # scheduler.step()
            # args.lr = optimizer.param_groups[0]['lr']
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
            
            # evaluate the model
            if args.dataset == 'custom':
                with num_torch_thread(1):
                    map, _ = test_for_train(_output_dir, model, args, val_path, names, device=device)
            else:
                map, _ = test_for_train(_output_dir, model, args)
            # save best model for update
            if map > max_map:
                with torch.no_grad():
                    best_model = copy.deepcopy(model)
                # map, _ = test_for_train(_output_dir, best_model, args, val_path, names, device=device)
                max_map = map
            
            # update on round completition
            if (epoch % args.epochPerRound == 0) or (epoch == (args.epochPerRound * args.max_rounds)+1):
                _round += 1
                with lock:
                    with torch.no_grad():
                        self.shared_memory[f'weightsClient_{clientId}'] = copy.deepcopy(best_model.to('cpu'))
                    self.shared_memory[f'updateClient_{clientId}']  = True
                if _round != args.max_rounds:
                    # wait for the update from server if it is not the last round
                    self.shared_memory[f'waitClient_{clientId}'] = True
                    while self.shared_memory[f'waitClient_{clientId}']:
                        time.sleep(0.2)
                        while  self.shared_memory[f'updateToClient_{clientId}']:
                            # get the aggregated model from the server for retraining
                            model = copy.deepcopy(self.shared_memory['aggr_model'].to('cpu'))
                            if args.use_cuda:
                                model.to(device)
                            if (ind < len(lr_sch)) and (_round == lr_sch[ind]):
                                ind += 1
                                args.lr = 0.1 * args.lr    
                            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
                            # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5,20,60,90,170,220,300], gamma=0.1)
                            model.train()    
                            self.shared_memory[f'waitClient_{clientId}']     = False
                            self.shared_memory[f'updateToClient_{clientId}'] = False
                else:
                    # update the client training completition to the server
                    self.shared_memory['Final_round'] = True                 
        print()

if __name__ == '__main__':
    Trainer()                                