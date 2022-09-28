from .OD_dataset import OD_Dataset
import os

def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    dataset_path = os.path.join(data_path, dataset_name)
    dataset = OD_Dataset(root=dataset_path, normal_class=normal_class)

    return dataset
