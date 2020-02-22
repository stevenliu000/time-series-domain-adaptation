import numpy as np
from torch.utils.data import Dataset

class TimeSeriesChunkDataset(Dataset):
    def __init__(self, x, y, context):
        super(TimeSeriesChunkDataset, self).__init__()
        self.x = x
        self.y = y
        self.context = context
        self.points_per_series = self.x.shape[1] - self.context + 1
        
    def __len__(self):
        return self.x.shape[0] * self.points_per_series

    def __getitem__(self, index):
        index_series = index // self.points_per_series
        index_point = index % self.points_per_series
        return_x = self.x[index_series, index_point:index_point+self.context, :]
        return_y = np.argmax(self.y[index_series, :])
        return return_x, return_y