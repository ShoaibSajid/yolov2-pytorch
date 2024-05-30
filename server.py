import os
import torch
import numpy as np
from Test_with_train import test_for_train
from util.data_util import check_dataset
import colorama
from colorama import Fore, Back, Style
import time
import copy

colorama.init(autoreset=True)

class Server:
    def __init__(self, shared_memory):
        super().__init__() 
        self.shared_memory =  shared_memory
        
    def aggregate(self, num_clients, Args, val_path, names, _round, device):
        aggr_model = copy.deepcopy(self.shared_memory['aggr_model'])
        # get all client models
        models = []
        mAPs = []
        for i in range(num_clients):
            out_dir = f'{Args.output_dir}/server/client_{i}'
            _model = copy.deepcopy(self.shared_memory[f'weightsClient_{i}'])
            models.append(_model)
            # mAPs.append(self.shared_memory[f'weightsClient_{i}'][1])
            _mAP, _ = test_for_train(out_dir, _model,
                                    Args, val_path,
                                    names, True,
                                    device=device)
            mAPs.append(_mAP)
            
        # Aggregate weights and update (to self.shared_memory['aggr_weights'])
        for name, param in aggr_model.named_parameters():
            print(f"Aggregating {name}")
            data= []
            for i,_m in enumerate(models):
                # mAP = (mAPs[i])*100
                # w   = mAP/mAP_sum
                _m_state = _m.state_dict()
                # _m_state = w * (_m_state[name].detach().cpu().numpy())
                data.append(_m_state[name].detach().cpu().numpy())
            aggr_data = np.mean(np.array(data), axis=0)
            param.data = torch.from_numpy(aggr_data)
            
        out_dir = f'{Args.output_dir}/server'
        Aggr_mAP, _ = test_for_train(out_dir, aggr_model.to('cpu'),
                                    Args, val_path,
                                    names, True,
                                    device=device)
        
        mAPs.append(Aggr_mAP)
        
        # log all results
        result = {'Round':_round}
        result['Server'] = mAPs[-1]
        for i in range(len(mAPs)-1):
            result[f'Client_{i}'] = mAPs[i]   
        # sweep.log(result)
        self.shared_memory['Results'] = result
        time.sleep(0.3)
        
        # find best model
        best_map = max(mAPs)
        _index   = mAPs.index(best_map)
        
        if _index==num_clients:
            best      = aggr_model
            save_name = os.path.join(out_dir, 'yolov2_server_best_aggr.pth')
        else: 
            best = models[_index]
            save_name = os.path.join(out_dir, f'yolov2_server_best_client{_index}.pth')
        # save best server model
        torch.save({
            'model': best.module.state_dict() if Args.mGPUs else best.state_dict(),
            }, save_name)
        
        return best
    
    def aggregate_weights(self, args, num_clients, device):
        _args = args
        # wdb   = args.wandb
        data_dict = check_dataset(_args.data)
        _, val_path, val_dir = data_dict['train'], data_dict['val'], data_dict['val_dir']
        _args.val_dir = val_dir
        names = data_dict['names']
        round = 0 
        # do until fedml is in progress
        while self.shared_memory[f'completition'] == False:
            time.sleep(0.2)
            # server waiting flag true until client training
            self.shared_memory[f'waitServer'] = True
            # parameter to check all client updates (stores True for client which have updated the new weights)
            check = [False for x in range(num_clients)]
            for i in range(num_clients):
                if self.shared_memory[f'updateClient_{i}']:
                    check[i] = True
            # check all clients have updated new weights to server or not
            if all(check):
                # if all clients have updated weights to server turn off server waiting to start aggragation
                self.shared_memory[f'waitServer'] = False         
            # start client validation, aggregation, aggregated model validation and best update to clients
            while (self.shared_memory[f'waitServer']==False) and (not self.shared_memory['Final_round']):
                best_ = self.aggregate(num_clients, _args, val_path, names, round, device)
                round += 1
                # update the best model to the clients
                self.shared_memory['aggr_model']  = copy.deepcopy(best_.to('cpu'))
                # update the updateToClient flag so that client 
                # can get the updated model and start training again
                for i in range(num_clients):
                    self.shared_memory[f'updateToClient_{i}'] = True
                    self.shared_memory[f'updateClient_{i}']   = False
                    self.shared_memory[f'weightsClient_{i}']  = None
                self.shared_memory[f'waitServer'] = True
            
            if self.shared_memory['Final_round']:
                # set completition status as True
                self.shared_memory[f'completition'] = True        
        
        while not self.shared_memory['stop']:
            time.sleep(0.2)
            # parameter to check all client updates (stores True for client which have updated the new weights)
            check = [False for x in range(num_clients)]
            for i in range(num_clients):
                if self.shared_memory[f'updateClient_{i}']:
                    check[i] = True
            # check all clients have updated new weights to server or not
            if all(check):
                # if all clients have updated weights to server turn off server waiting to start aggragation
                self.shared_memory['stop'] = True
        
        # no need to update at the last round
        best_ = self.aggregate(num_clients, _args, val_path, names, round, device)
        # self.shared_memory['aggr_model'] = best_.to('cpu')
            
        print(f'{Fore.GREEN}{Style.BRIGHT}Federated rounds complete aborting...')
        self.shared_memory['kill'] = True

if __name__ == '__main__':
    Server()                       
                    
                        