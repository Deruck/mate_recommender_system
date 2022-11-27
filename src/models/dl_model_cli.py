from pydantic import FilePath

from data_module import BaseDataModule
from data_module.entities import DateList, UserDict

from model_cli_base import BaseModelCli
from consts import MODEL
from dl_models import get_dl_model
from model_arguments import DL_ModelArgs



class DLModelCli(BaseModelCli):
    
    def __init__(self, model: MODEL):
        self._model_name = model.value
        self._model = get_dl_model(model)
        
    def train(self, model_args: DL_ModelArgs, data_module: BaseDataModule):
        return super().train(model_args, data_module)
    
    def load_model(self, model_save_path: FilePath):
        return super().load_model(model_save_path)
    
    def inference(self, date_list: DateList, user_dict: UserDict) -> DateList:
        return super().inference(date_list, user_dict)