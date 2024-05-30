from train_yolov2_tiny import train
import tqdm
import torch
import time
from config import config as cfg


def num_torch_thread(n_thread: int):
    n_thread_original = torch.get_num_threads()
    torch.set_num_threads(n_thread)
    yield
    torch.set_num_threads(n_thread_original)

class Trainer:
    def __init__(self, shared_memory):
        super().__init__()
        
        self.shared_memory =  shared_memory
        
    def train_client(self, args, clientId, device, lock):
        # set up the necessary args for client training
        _args = args
        _args.data       = _args.client_data[clientId]
        _args.max_epochs = _args.epochPerRound
        _args.fedml      = True
        _args.output_dir = f'{_args.output_dir}/client_{clientId}'
        _args.model      = self.shared_memory['aggr_model'].to('cpu')
        _args.lr         = cfg.lr
        
        for round in tqdm.tqdm(range(_args.max_rounds)):
            
            if round == _args.max_rounds-1:
                # update the client training completition to the server
                self.shared_memory['Final_round'] = True
            
            if (round>0) & (round<_args.max_rounds-1):
                # use the new model from server for retraining
                _args.lr     *= ((1/round) * 0.2)
                _args.weights = ''
                _args.resume  = False
            
            # get the best model of the client after each round
            best_model, map = train(_args, device)
            best_model = best_model.to('cpu')
            
            with lock:
                self.shared_memory[f'weightsClient_{clientId}'] = (best_model,map)
            self.shared_memory[f'updateClient_{clientId}']  = True
                
            
            if round != _args.max_rounds-1:
                # wait for the update from server if it is not the last round
                self.shared_memory[f'waitClient_{clientId}'] = True
                while self.shared_memory[f'waitClient_{clientId}']:
                    time.sleep(0.2)
                    while  self.shared_memory[f'updateToClient_{clientId}']:
                        # get the aggregated model from the server for retraining
                        _args.model = self.shared_memory['aggr_model'].to('cpu')
                        self.shared_memory[f'waitClient_{clientId}']     = False
                        self.shared_memory[f'updateToClient_{clientId}'] = False         
            
        print()

if __name__ == '__main__':
    Trainer()        