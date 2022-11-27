from abc import ABCMeta, abstractproperty
from sklearn.base import ClassifierMixin
from typing import Type, Dict, List, Protocol
from xgboost import XGBClassifier
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
            'eta': 0.3,
            'gamma': 0, 
            'min_child_weight': 1,
            'colsample_bytree': 1, 
            'colsample_bylevel': 1, 
            'subsample': 1, 
            'reg_lambda': 1, 
            'reg_alpha': 0,
            'seed': 33
        }
        
    @property
    def gs_params(self) -> Dict[str, List[float]]:
        return {
            'n_estimators': [10, 50, 100, 200, 500], 
            'max_depth': [2, 3, 5, 8, 15], 
        }
        
    
#############################################################################################
## Apis
#############################################################################################

def get_sklearn_model(model: MODEL) -> BaseSKLModel:
    if model == MODEL.XGB:
        return XGBoost()
    else:
        raise Exception("model not supported")