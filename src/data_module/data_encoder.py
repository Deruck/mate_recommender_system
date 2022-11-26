from abc import ABCMeta, abstractmethod
import numpy as np
from typing import Set, Literal, Dict
from pydantic import BaseModel

from entities import User, UserDict, UserId, UserDict, CATEGORICAL_VARIABLES, NUMERICAL_VARIABLES

class BaseDataEncoder(metaclass=ABCMeta):
    """数据编码"""
    
    @abstractmethod
    def fit_transform(self, user_dict: UserDict) -> UserDict: ...
    
    @abstractmethod
    def transform(self, user_dict: UserDict) -> UserDict: ...

class MeanStdPair(BaseModel):
    mean: float
    std: float

class ZScoreDataEncoder(BaseDataEncoder):
    
    def __init__(self) -> None:
        self.__variable_mean_std_dict: Dict[str, MeanStdPair] = {}
    
    def fit_transform(self, user_dict: UserDict) -> UserDict:
        # 计算放缩scale
        for variable in NUMERICAL_VARIABLES:
            values = np.array([user.dict()[variable] for user in user_dict.values()])
            self.__variable_mean_std_dict[variable] = MeanStdPair(
                mean = values.mean(),
                std = values.std()
            )
        return self.transform(user_dict)
            
    def transform(self, user_dict: UserDict) -> UserDict:
        for user in user_dict.values():
            user_dict_form = user.dict()
            for variable in NUMERICAL_VARIABLES:
                user_dict_form[variable] = (user_dict_form[variable] - self.__variable_mean_std_dict[variable].mean) / self.__variable_mean_std_dict[variable].std
                user = User(**user_dict_form)
        return user_dict
    

