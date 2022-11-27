from abc import ABCMeta, abstractmethod
from pydantic import FilePath, DirectoryPath
from typing import List

from data_module import ZScoreDataModule, BaseDataModule
from data_module.entities import UserDict, DateList
from utils import LoggerManager, BaseArguments


logger = LoggerManager.get_logger()

class BaseModelArgs(BaseArguments):
    ...

class BaseModelCli(metaclass=ABCMeta):
    """模型api"""
    @abstractmethod
    def train(self, model_args: BaseModelArgs, data_module: BaseDataModule, model_save_dir: DirectoryPath): ...
        
    @abstractmethod
    def load_model(self, model_save_dir: DirectoryPath): ...
    
    @abstractmethod
    def inference(self, date_list: DateList, user_dict: UserDict) -> DateList: ...
    
