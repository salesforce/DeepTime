import os
from os.path import join
from typing import Optional, List, Tuple

import gin
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from utils.time_features import get_time_features


@gin.configurable()
class ForecastDataset(Dataset):
    def __init__(self,
                 flag: str,
                 horizon_len: int,
                 scale: bool,
                 cross_learn: bool,
                 data_path: str,
                 root_path: Optional[str] = 'storage/datasets',
                 features: Optional[str] = 'S',
                 target: Optional[str] = 'OT',
                 lookback_len: Optional[int] = None,
                 lookback_aux_len: Optional[int] = 0,
                 lookback_mult: Optional[float] = None,
                 time_features: Optional = [],
                 normalise_time_features: Optional = True,):
        """
        :param flag: train/val/test flag
        :param horizon_len: number of time steps in forecast horizon
        :param scale: performs standard scaling
        :param data_path: relative (to root_path) path to data file (.csv)
        :param cross_learn: treats multivariate time series as multiple univar time series
        :param root_path: path to datasets folder
        :param features: multivar (M) or univar (S) forecasting
        :param target: name of target variable for univar forecasting (features=S)
        :param lookback_len: number of time steps in lookback window
        :param lookback_aux_len: number of time steps to append to y from the lookback window
        (for models with decoders which requires initialisation)
        :param lookback_mult: multiplier to decide lookback window length
        """
        assert flag in ('train', 'val', 'test'), \
            f"flag should be one of (train, val, test)"
        assert features in ('M', 'S'), \
            f"features should be one of (M: multivar, S: univar)"
        assert (lookback_len is not None) ^ (lookback_mult is not None), \
            f"only 'lookback_len' xor 'lookback_mult' should be specified"

        self.flag = flag
        sself.lookback_len = int(horizon_len * lookback_mult) if lookback_mult is not None else lookback_len
        self.lookback_aux_len = lookback_aux_len
        self.horizon_len = horizon_len
        self.scale = scale
        self.cross_learn = cross_learn
        self.data_path = data_path
        self.root_path = root_path
        self.features = features
        self.target = target
        self.time_features = time_features
        self.normalise_time_features = normalise_time_features

        self.n_dims = None
        self.scaler = None
        self.data_x = None
        self.data_y = None
        self.timestamps = None
        self.n_time = None
        self.n_time_samples = None
        self.load_data()

    def load_data(self):
        df_raw = pd.read_csv(join(self.root_path, self.data_path))  # (n_obs, date + n_feats)
        cols = list(df_raw.columns)
        cols.remove('date')
        cols.remove(self.target)
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1s, border2s, border1, border2 = self.get_borders(df_raw)

        if self.features == 'M':
            df_data = df_raw[cols + [self.target]]
            self.n_dims = len(cols + [self.target])
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            self.n_dims = 1
        else:
            raise ValueError

        self.scaler = StandardScaler()
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.timestamps = get_time_features(pd.to_datetime(df_raw.date[border1:border2].values),
                                            normalise=self.normalise_time_features,
                                            features=self.time_features)
        self.n_time = len(self.data_x)
        self.n_time_samples = self.n_time - self.lookback_len - self.horizon_len + 1

    def get_borders(self, df_raw: pd.DataFrame) -> Tuple[List[int], List[int], List[int], List[int]]:
        set_type = {'train': 0, 'val': 1, 'test': 2}[self.flag]
        if self.data_path.startswith('ETT-small/ETTh'):
            border1s = [0, 12 * 30 * 24 - self.lookback_len, 12 * 30 * 24 + 4 * 30 * 24 - self.lookback_len]
            border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        elif self.data_path.startswith('ETT-small/ETTm'):
            border1s = [0, 12 * 30 * 24 * 4 - self.lookback_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.lookback_len]
            border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        else:
            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_val = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.lookback_len, len(df_raw) - num_test - self.lookback_len]
            border2s = [num_train, num_train + num_val, len(df_raw)]
        border1 = border1s[set_type]
        border2 = border2s[set_type]
        return border1s, border2s, border1, border2

    def __len__(self):
        if self.cross_learn:
            return self.n_time_samples * self.n_dims
        return self.n_time_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.cross_learn:
            dim_idx = idx // self.n_time_samples
            dim_slice = slice(dim_idx, dim_idx + 1)
            idx = idx % self.n_time_samples
        else:
            dim_slice = slice(None)

        x_start = idx
        x_end = x_start + self.lookback_len
        y_start = x_end - self.lookback_aux_len
        y_end = y_start + self.lookback_aux_len + self.horizon_len

        x = self.data_x[x_start:x_end, dim_slice]
        y = self.data_y[y_start:y_end, dim_slice]
        x_time = self.timestamps[x_start:x_end]
        y_time = self.timestamps[y_start:y_end]

        return x, y, x_time, y_time

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)