import numpy as np
import h5py
import torch

from torch.utils.data import Dataset


class TimeSeriesDataSet(Dataset):
    def __init__(self, features: torch.tensor, labels: torch.tensor,
                 train_size: int, valid_size: int, skip_size: int):
        """
        DataSet to operate over Time Series data

        Args:
            features: torch tensor of size [n_features, seq_len]
            labels: torch tensor of size [n_features, seq_len]
            train_size: number of time steps in TRAIN slice
            valid_size: number of time steps in VALID slice
            skip_size: number of time steps to skip between TRAIN and VALID slices
        """
        self.features = features.t().contiguous()
        self.labels = labels
        self.train_size = train_size
        self.valid_size = valid_size
        self.skip_size = skip_size

        self.tot_size = train_size + skip_size + valid_size
        self.offset = train_size + skip_size

        # add categorical labels
        self.id_to_value = self.labels.unique(sorted=True).tolist()
        self.value_to_id = {v: i for i, v in enumerate(self.id_to_value)}
        self.cat_labels = self.get_categorical_labels()

        assert len(self) > 0, f'Invalid sizing {self.features.size(1)} < {self.tot_size}!'

    def get_categorical_labels(self):
        """Maps continuous values to their categorical representation
           Allows the training as a classifier instead of a regressor
        """
        np_cat_labels = np.vectorize(self.value_to_id.__getitem__)(self.labels.numpy())
        return torch.from_numpy(np_cat_labels).long()

    @staticmethod
    def load_matlab(mat_path: str, size: int = None):
        series = {}
        with h5py.File(mat_path, 'r') as f:
            for name, data in f.items():
                print(f'Loading: {name} ...')
                # replace NaN with 0
                np_data = np.nan_to_num(np.array(data))
                series[name] = np_data

        # retrieve labels
        labels = series.pop('y')

        # concatenate all features (15x4)
        features = np.concatenate(list(series.values()), axis=0)

        # truncate?
        if size is not None:
            print(f'Truncating sequence to length {size}')
            features = features[:, :size]
            labels = labels[:, :size]

        # numpy -> torch
        t_features = torch.from_numpy(features).float()
        t_labels = torch.from_numpy(labels).float().squeeze()

        return t_features, t_labels

    def __len__(self):
        return self.features.size(0) - self.tot_size + 1

    def __getitem__(self, i):
        if not 0 <= i < len(self):
            raise IndexError(f'index {i} is out of bounds for dataset of size {len(self)}')

        return dict(features=self.features[i:i + self.tot_size, :],
                    labels=self.labels[i + self.offset:i + self.tot_size],
                    cat_labels=self.cat_labels[i + self.offset:i + self.tot_size])

    @property
    def num_features(self):
        return self.features.size(1)

    @property
    def num_labels(self):
        return len(self.id_to_value)

    @classmethod
    def from_matlab(cls, mat_path: str, train_size: int, valid_size: int, skip_size: int, size: int = None):
        """
            Loads a .mat file containing order book data series and instantiate a TimeSeriesDataSet

            Args:
                mat_path: path to Matlab file
                ...
                size: max size to truncate (useful for CV)
        """
        features, labels = TimeSeriesDataSet.load_matlab(mat_path, size)
        return cls(features, labels, train_size, valid_size, skip_size)
