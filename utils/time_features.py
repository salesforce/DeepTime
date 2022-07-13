from abc import ABC, abstractmethod
from typing import Optional, List, Union

import numpy as np
import pandas as pd


class TimeFeature(ABC):
    """Abstract class for time features"""
    def __init__(self, normalise: bool, a: float, b: float):
        self.normalise = normalise
        self.a = a
        self.b = b

    @abstractmethod
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def _max_val(self) -> float:
        ...

    @property
    def max_val(self) -> float:
        return self._max_val if self.normalise else 1.0

    def scale(self, val: np.ndarray) -> np.ndarray:
        return val * (self.b - self.a) + self.a

    def process(self, val: np.ndarray) -> np.ndarray:
        features = self.scale(val / self.max_val)
        if self.normalise:
            return features
        return features.astype(int)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(normalise={self.normalise}, a={self.a}, b={self.b})"


class SecondOfMinute(TimeFeature):
    """Second of minute, unnormalised: [0, 59]"""
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return self.process(idx.second)

    @property
    def _max_val(self):
        return 59.0


class MinuteOfHour(TimeFeature):
    """Minute of hour, unnormalised: [0, 59]"""
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return self.process(idx.minute)

    @property
    def _max_val(self):
        return 59.0


class HourOfDay(TimeFeature):
    """Hour of day, unnormalised: [0, 23]"""
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return self.process(idx.hour)

    @property
    def _max_val(self):
        return 23.0


class DayOfWeek(TimeFeature):
    """Hour of day, unnormalised: [0, 6]"""
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return self.process(idx.dayofweek)

    @property
    def _max_val(self):
        return 6.0


class DayOfMonth(TimeFeature):
    """Day of month, unnormalised: [0, 30]"""
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return self.process(idx.day - 1)

    @property
    def _max_val(self):
        return 30.0


class DayOfYear(TimeFeature):
    """Day of year, unnormalised: [0, 365]"""
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return self.process(idx.dayofyear - 1)

    @property
    def _max_val(self):
        return 365.0


class WeekOfYear(TimeFeature):
    """Week of year, unnormalised: [0, 52]"""
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return self.process(pd.Index(idx.isocalendar().week, dtype=int) - 1)

    @property
    def _max_val(self):
        return 52.0

class MonthOfYear(TimeFeature):
    """Month of year, unnormalised: [0, 11]"""
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return self.process(idx.month - 1)

    @property
    def _max_val(self):
        return 11.0


class QuarterOfYear(TimeFeature):
    """Quarter of year, unnormalised: [0, 3]"""
    def __call__(self, idx: pd.DatetimeIndex) -> np.ndarray:
        return self.process(idx.quarter - 1)

    @property
    def _max_val(self):
        return 3.0


str_to_feat = {
    # dictionary mapping name to TimeFeature function
    'SecondOfMinute': SecondOfMinute,
    'MinuteOfHour': MinuteOfHour,
    'HourOfDay': HourOfDay,
    'DayOfWeek': DayOfWeek,
    'DayOfMonth': DayOfMonth,
    'DayOfYear': DayOfYear,
    'WeekOfYear': WeekOfYear,
    'MonthOfYear': MonthOfYear,
    'QuarterOfYear': QuarterOfYear,
}


freq_to_feats = {
    # dictionary mapping frequency to list of TimeFeature functions
    'q': [QuarterOfYear],
    'm': [QuarterOfYear, MonthOfYear],
    'w': [QuarterOfYear, MonthOfYear, WeekOfYear],
    'd': [QuarterOfYear, MonthOfYear, WeekOfYear, DayOfYear, DayOfMonth, DayOfWeek],
    'h': [QuarterOfYear, MonthOfYear, WeekOfYear, DayOfYear, DayOfMonth, DayOfWeek, HourOfDay],
    't': [QuarterOfYear, MonthOfYear, WeekOfYear, DayOfYear, DayOfMonth, DayOfWeek, HourOfDay, MinuteOfHour],
    's': [QuarterOfYear, MonthOfYear, WeekOfYear, DayOfYear, DayOfMonth, DayOfWeek, HourOfDay, MinuteOfHour, SecondOfMinute],
}


def get_time_features(dates: pd.DatetimeIndex, normalise: bool, a: Optional[float] = 0., b: Optional[float] = 1.,
                      features: Optional[Union[str, List[str]]] = None) -> np.ndarray:
    """
    Returns a numpy array of date/time features based on either frequency or directly specifying a list of features.
    :param dates: DatetimeIndex object of shape (time,)
    :param normalise: Whether to normalise feature between [a, b]. If not, return as an int in the original feature range.
    :param a: Lower bound of feature
    :param b: Upper bound of feature
    :param features: Frequency string used to obtain list of TimeFeatures, or directly a list of names of TimeFeatures
    :return: np array of date/time features of shape (time, n_feats)
    """
    if isinstance(features, list):
        assert all([feat in str_to_feat.keys() for feat in features]), \
            f"items in list should be one of {[*str_to_feat.keys()]}"
        features = [str_to_feat[feat] for feat in features]
    elif isinstance(features, str):
        assert features in freq_to_feats.keys(), \
            f"features should be one of {[*freq_to_feats.keys()]}"
        features = freq_to_feats[features]
    else:
        raise ValueError(f"features should be a list or str, not a {type(features)}")

    features = [feat(normalise, a, b)(dates) for feat in features]

    if len(features) == 0:
        return np.empty((dates.shape[0], 0))
    return np.stack(features, axis=1)
