from torch.utils.data import Dataset, DataLoader


class JoinDataset(Dataset):
    def __init__(self, source_x, source_y, target_x, target_y, random=False):
        self.source_x = source_x
        self.source_y = source_y
        self.target_x = target_x
        self.target_y = target_y
        
        self.source_len = self.source_y.shape[0]
        self.target_len = self.target_y.shape[0]
    
        self.random = random
    def __len__(self):
        return self.target_len
    
    def __getitem__(self, index):
        if self.random:
            index_source = random.randrange(source_len)
            index_target = random.randrange(target_len)
        else:
            index_source = index
            index_target = index

        return (self.source_x[index_source], self.source_y[index_source]), (self.target_x[index_target], self.target_y[index_target])
    
    
class SingleDataset(Dataset):
    def __init__(self, x, y):
            self.x = x
            self.y = y
            self.len = self.y.shape[0]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
