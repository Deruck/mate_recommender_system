from pydantic import FilePath, DirectoryPath
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Literal, List
import numpy as np
import shutil
import os
import re

from data_module import BaseDataModule
from data_module.entities import DateList, UserDict

from model_cli_base import BaseModelCli
from consts import MODEL
from dl_models import DNN, EBDDNN, FORK
from model_arguments import DL_ModelArgs
from utils import LoggerManager

logger = LoggerManager.get_logger()


class DLModelCli(BaseModelCli):
    
    def __init__(self, model: MODEL):
        self._model_name: str = model.value
        self._model = model
        
    def train(self, model_args: DL_ModelArgs, data_module: BaseDataModule, model_save_dir):
        logger.info(f"training model: {self._model_name}")
        model_save_dir = model_save_dir / self._model_name
        if not model_save_dir.exists():
            model_save_dir.mkdir()
        self.__clean_save_dir(model_save_dir)
        tb_logger = TensorBoardLogger(
            save_dir=model_save_dir,
            name="tensor_board_logs",
            max_queue=2, 
            flush_secs=5
        )
        ckpt = ModelCheckpoint(
            monitor="dev_acc",
            mode="max", 
            save_top_k=1,
            save_on_train_epoch_end=False, 
            verbose=True,
            dirpath=model_save_dir,
            filename=f"{self._model_name}",
        )
        trainer = Trainer(
            accelerator="gpu", 
            devices=1, 
            callbacks=[ckpt],
            max_epochs=model_args.epochs,
            gradient_clip_val=0.25,
            default_root_dir=model_save_dir / self._model_name,
            enable_progress_bar=False,
            auto_select_gpus=True,
            logger=[tb_logger]
        )
        if self._model == MODEL.DNN:
            model = DNN(
                model_args,
                feature_num=data_module.feature_num
            )
        elif self._model == MODEL.EBDDNN:
            model = EBDDNN(
                model_args,             # type: ignore
                data_module.cate_feature_dict,
                data_module.feature_num
            )
        elif self._model == MODEL.FORK:
            model = FORK(
                model_args,             # type: ignore
                data_module.cate_feature_dict,
                data_module.feature_num
            )
        else:
            raise Exception("model not supported")
            
        trainer.fit(
            model=model, 
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader()
        )
        test_true_label = [date.dec for date in data_module.test_date_list]
        output: List[Literal[0, 1]] = trainer.predict( # type: ignore
            model=model,
            dataloaders=data_module.test_dataloader()
        )
        test_acc = (np.array(test_true_label) == np.array(output)).mean()
        logger.info(f"test_acc: {test_acc:.4f}")
        log_file = LoggerManager.get_log_file()
        shutil.copy(log_file, model_save_dir / "train_log.log")
        
    def __clean_save_dir(self, model_save_dir: DirectoryPath):
        file_list = os.listdir(model_save_dir)
        ckpt_file_list = [model_save_dir / file for file in file_list if re.match(r"^.*\.ckpt$", file) is not None]
        for ckpt in ckpt_file_list:
            os.remove(ckpt)
    
    def load_model(self, model_save_path: FilePath):
        return super().load_model(model_save_path)
    
    def inference(self, date_list: DateList, user_dict: UserDict) -> DateList:
        return super().inference(date_list, user_dict)