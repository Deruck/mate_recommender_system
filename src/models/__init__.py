import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from typing import Tuple

from consts import MODEL
from model_cli_base import BaseModelCli, BaseModelArgs
from dl_model_cli import DLModelCli
from sklearn_model_cli import SKLModelCli
from model_arguments import DL_ModelArgs, SK_ModelArgs, DNNModelArgs, EBDDNN_ModelArgs, FORK_ModelArgs
    
def model_cli_factory(model: MODEL) -> Tuple[BaseModelCli, BaseModelArgs]:
    if model == MODEL.DNN:
        return (DLModelCli(model), DNNModelArgs())
    if model == MODEL.EBDDNN:
        return (DLModelCli(model), EBDDNN_ModelArgs())
    if model == MODEL.FORK:
        return (DLModelCli(model), FORK_ModelArgs())
    elif model in [MODEL.XGB]:
        return (SKLModelCli(model), SK_ModelArgs())
    else:
        raise Exception("model not supported")