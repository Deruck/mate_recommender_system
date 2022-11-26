from enum import Enum

class MODEL(Enum):
    DNN = "dnn"
    XGB = "xgboost"
    LR = "lr"
    
class DL_MODEL(Enum):
    DNN = "dnn"
    
class SK_MODEL(Enum):
    XGB = "XGBoost"
    LR = "Logistic_Regression"