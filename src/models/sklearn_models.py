from abc import ABCMeta, abstractproperty
from sklearn.base import ClassifierMixin
from typing import Type, Dict, List, Protocol
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

from consts import MODEL

#############################################################################################
## Base
#############################################################################################

class ClassifierProto(Protocol):
    def fit(self, *args, **kwargs) -> None: ...
    def predict(self, *args, **kwargs) -> np.ndarray: ...
    def predict_proba(self, *args, **kwargs) -> np.ndarray: ...


class BaseSKLModel(metaclass=ABCMeta):
    
    @abstractproperty
    def model_cls(self) -> Type[ClassifierProto]: ...
    
    @abstractproperty
    def other_params(self) -> Dict[str, float]: ...
    
    @abstractproperty
    def gs_params(self) -> Dict[str, List[float]]: ...
    
#############################################################################################
## Subs
#############################################################################################


class XGBoost(BaseSKLModel):
    
    @property
    def model_cls(self) -> Type[ClassifierMixin]:
        return XGBClassifier
    
    @property
    def other_params(self) -> Dict[str, float]:
        return {
            'seed': 33
        }
        
    @property
    def gs_params(self) -> Dict[str, List[float]]:
        return {
            'n_estimators': [5000, 5100, 5200, 5300], 
            'max_depth': [5],
            "gamma": [0],
            "eta": [0.2, 0.1, 0.05, 0.01],
            "lambda": [1.1, 1.15, 1.2],
            "subsample": [0.85]
        }
        
class LR(BaseSKLModel):
    
    @property
    def model_cls(self) -> Type[ClassifierMixin]:
        return LogisticRegression
    
    @property
    def other_params(self) -> Dict[str, float]:
        return {
            'n_jobs': -1,
            "verbose": 0,
        }
        
    @property
    def gs_params(self) -> Dict[str, List]:
        return {
            "penalty": ["l1", "l2", "none"],
            "C": [1, 1.2, 1.4]
        }
    
#############################################################################################
## APIs
#############################################################################################

def get_sklearn_model(model: MODEL) -> BaseSKLModel:
    if model == MODEL.XGB:
        return XGBoost()
    elif model == MODEL.LR:
        return LR()
    else:
        raise Exception("model not supported")