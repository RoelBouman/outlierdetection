from torch.utils.data import Dataset
import pickle
import numpy as np
from sklearn.preprocessing import RobustScaler

from base.torchvision_dataset import TorchvisionDataset

class OD_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)


        self.root = root
    
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = [0,1]
        self.outlier_classes.remove(normal_class)

        # Subset train set to normal class
        self.train_set = OD_Base_Dataset(dataset_name=root)

        self.test_set = OD_Base_Dataset(dataset_name=root)

class OD_Base_Dataset(Dataset):
    def __init__(self, dataset_name):
        
        data = pickle.load(open(dataset_name, 'rb'))
        self.X, self.y = data["X"].astype(np.float32), np.squeeze(data["y"]).astype(np.float32)
        
        scaler = RobustScaler()
        
        self.X_scaled = scaler.fit_transform(self.X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

            
        return self.X_scaled[idx,:], self.y[idx], idx
