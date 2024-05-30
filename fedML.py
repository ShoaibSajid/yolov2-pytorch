import os
import time
import torch
import multiprocessing
from multiprocessing import Process, Manager, freeze_support
from flags import Flags
import argparse
from util.data_util import check_dataset
import shutil
import yaml
# import ruamel.yaml as yaml1
from trainer_new import Trainer
from server import Server
from yolov2_tiny_2 import Yolov2
import wandb

# multiprocessing.set_start_method('forkserver')

def wandbSweep():
    os.environ["WANDB_API_KEY"] = 'de330e5cd3019a36af5494ebdd8b8ce4a19161c3'
    return wandb.init(project='Federated_Training')

# print(multiprocessing.get_start_method())  

class Fedml:
    def __init__(self, num_clients):
        super().__init__()
        
        # Start shared memory manager server
        # Create event based manager to help in system shutdown
        manager = Manager()
        self.shared_memory        =  manager.dict()
        self.lock                 =  manager.Lock()
        # self.shared_memory_server =  manager.Event()

        # number of clients to be passes by user
        self.num_clients  =  num_clients
        self.aggr_weights = []
        
        # necessary control flags
        self.Flags       =  Flags(self.shared_memory, self.num_clients)
        # server for training coordination
        self.server      =  Server(self.shared_memory)
        # trainer trains individul clients
        self.trainer     =  Trainer(self.shared_memory)
        
    def split_data(self):
        '''
        split the training data between clients
        
        '''
        # make new (every time) directory to store client data
        if not os.path.exists(f'{os.getcwd()}/fedmldata'):
            os.makedirs(f'{os.getcwd()}/fedmldata')
        else:
            shutil.rmtree(f'{os.getcwd()}/fedmldata', ignore_errors=True)
            os.makedirs(f'{os.getcwd()}/fedmldata')    
        data_dict = check_dataset(args.data)
        train_path, val_path, val_dir = data_dict['train'], data_dict['val'], data_dict['val_dir']
        # list to hold each client data file path
        data_paths = []
        with open(train_path, 'r') as f:
            lines = f.readlines()
            # split full data in all the clients
            dev   = len(lines)/self.num_clients
            start = 0
            end   = dev
            for i in range(self.num_clients):
                _data  = dict()
                _lines = lines[int(start):int(end)]
                # make train.txt (txt file containing path lists --> yolo style) file for ith client
                with open(f'{os.getcwd()}/fedmldata/client_{i}.txt', 'w+') as f1:
                    f1.writelines(_lines)
                start = end
                end = end + dev
                
                # make yaml file for ith client
                names = data_dict['names']
                _data['path' ]    =  data_dict['path']
                _data['train']    =  f'{os.getcwd()}/fedmldata/client_{i}.txt'
                _data['val'  ]    =  data_dict['val']
                _data['val_dir']  =  data_dict['val_dir']
                _data['nc'   ]    =  data_dict['nc']
                _data['names']    =  names
                with open(f'{os.getcwd()}/fedmldata/client_{i}.yaml', 'w+') as file:
                    yaml.dump(_data,file)
                data_paths.append(f'{os.getcwd()}/fedmldata/client_{i}.yaml')                            
            print()
        return data_paths, data_dict['names']
    
    def main(self, args):
        #wandb logging
        run = wandbSweep()
        # split training data between clients (Homogeneous data mode)
        data_, names = self.split_data()
        # load model
        aggr_model = Yolov2(classes=names)
        # load initial server weights
        pre_trained_checkpoint = torch.load(args.weights,map_location='cpu')
        aggr_model.load_state_dict(pre_trained_checkpoint['model'])
        # update server model in the server memory
        self.shared_memory['aggr_model']  = aggr_model.to('cpu')
        # each client data for training
        args.client_data = data_
        # args.wandb = wdb
        # start server process
        devices  = args.device
        #server process
        server   = Process(target=self.server.aggregate_weights, args=(args, self.num_clients, devices[0]))
        # client 0
        client_0 = Process(target=self.trainer.train_client, args=(args, 0, devices[1], self.lock))
        
        #client 1
        client_1 = Process(target=self.trainer.train_client, args=(args, 1, devices[2], self.lock))
        
        #client 2
        client_2 = Process(target=self.trainer.train_client, args=(args, 2, devices[3], self.lock))
        
        # #client 3
        client_3 = Process(target=self.trainer.train_client, args=(args, 3, devices[4], self.lock))
        
        server.start()
        client_0.start()
        client_1.start()
        client_2.start()
        client_3.start()
        
        while not self.shared_memory['kill']:
            time.sleep(0.3)
            if len(self.shared_memory['Results']) > 0:
                run.log(self.shared_memory['Results'])
                self.shared_memory['Results'] = []
        
        server.join()
        client_0.join()
        client_1.join()
        client_2.join()
        client_3.join()
        print('done')
        # add processes  

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Yolo v2')
    parser.add_argument('--max_rounds', dest='max_rounds',
                        help='number of Federated rounds',
                        default=3, type=int)
    parser.add_argument('--epochPerRound', type=int,
                        default=2,
                        help='number of epochs per Federated round')
    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=1, type=int)
    parser.add_argument('--batch_size', dest='batch_size',
                        default=4, type=int)
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
    parser.add_argument('--weights', default='', dest='weights',
                        help='provide the path of weight file (.pth) if resume')
    parser.add_argument('--exp_name', dest='exp_name',
                        default='default', type=str)
    parser.add_argument('--device', type= list,
                        default=[0,1,2,3,4], dest='device',
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
    parser.add_argument('--numClients', type=int,
                         default=4,
                         help='number of federated clients')
    args = parser.parse_args()
    return args

# python fedML.py --output_dir /home/gpuadmin/ZPD/yolov2-pytorch/output/Bdd_fedml --datase custom --data data_Bdd.yaml --weights /home/gpuadmin/ZPD/yolov2-pytorch/output/Bdd_3/yolov2_best_map.pth

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    args = parse_args()
    fedMl = Fedml(args.numClients)
    fedMl.main(args)
    print('done')