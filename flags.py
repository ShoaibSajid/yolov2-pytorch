class Flags:
    def __init__(self, shared_memory, num_clients):
        super().__init__()
                
        # Create gui under shared_memory
        self.shared_memory = shared_memory
        
        self.shared_memory['names'] = []
        
        for i in range(num_clients):
            self.shared_memory[f'updateClient_{i}']   = False
            self.shared_memory[f'weightsClient_{i}']  = None
            self.shared_memory[f'updateToClient_{i}'] = False
            self.shared_memory[f'waitClient_{i}']     = False
            
        self.shared_memory[f'waitServer']   = False
        self.shared_memory['aggr_model']    = None
        self.shared_memory['Final_round']   = False
        self.shared_memory[f'completition'] = False
        self.shared_memory['stop']          = False
        self.shared_memory['kill']          = False      
        self.shared_memory['Results']       = []