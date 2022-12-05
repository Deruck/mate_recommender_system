from pydantic import FilePath, DirectoryPath
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Literal, List, Dict, Any, Type
import numpy as np
import shutil
import os
import re
from torch.utils.data import DataLoader
from pathlib import Path
import json

from data_module import BaseDataModule
from data_module.entities import DateList, UserDict

from model_cli_base import BaseModelCli
from consts import MODEL
from dl_models import DNN, EBDDNN, FORK, TEST, BaseModel
from model_arguments import DL_ModelArgs
from utils import LoggerManager, evaluate_model
from data_module.datasets import InfDataset


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
        model = self.__init_model(model_args ,data_module.feature_num, data_module.cate_feature_dict)
        trainer.fit(
            model=model, 
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=data_module.val_dataloader()
        )
        test_true_label: List[int] = [date.dec for date in data_module.test_date_list] # type: ignore
        model = model.load_from_checkpoint(model_save_dir / f"{self._model_name}.ckpt", model_args=model_args, feature_num=data_module.feature_num, cate_feature_dict=data_module.cate_feature_dict)
        output: List[float] = trainer.predict( # type: ignore
            model=model,
            dataloaders=data_module.test_dataloader()
        )
        evaluate_model(output, test_true_label)
        
        log_file = LoggerManager.get_log_file()
        shutil.copy(log_file, model_save_dir / "train_log.log")
        hyper_params = {
            "feature_num": data_module.feature_num,
            "cate_feature_dict": data_module.cate_feature_dict
        }
        json.dump(hyper_params, open(model_save_dir / "hyper_params.json", "w", encoding="utf-8"), indent=2)
    
    def load_model(self, model_args: DL_ModelArgs, model_save_dir: DirectoryPath):
        model_save_dir = model_save_dir / self._model_name
        hyper_params: Dict[str, Any] = json.load(open(model_save_dir / "hyper_params.json", "r", encoding="utf-8"), parse_int=int)
        hyper_params["cate_feature_dict"] = {int(k): v for k, v in hyper_params["cate_feature_dict"].items()}
        model_cls_dict: Dict[MODEL, Type[BaseModel]] = {
            MODEL.DNN: DNN,
            MODEL.EBDDNN: EBDDNN,
            MODEL.FORK: FORK
        }
        self._inf_model = model_cls_dict[self._model].load_from_checkpoint(model_save_dir / f"{self._model.value}.ckpt", strict=True, model_args=model_args, **hyper_params)
    
    def inference(self, date_list: DateList, user_dict: UserDict) -> DateList:
        dataloader = DataLoader(InfDataset(date_list, user_dict), batch_size=512)
        self._trainer =  Trainer(
            accelerator="gpu",
            devices=1,
            default_root_dir=str(Path.cwd() / "tmp"),
            enable_progress_bar=False,
            auto_select_gpus=True
        )
        output: List[float] = self._trainer.predict( # type: ignore
            model=self._inf_model,
            dataloaders=dataloader
        )
        for date, dec in zip(date_list, output):
            date.dec = dec
        return date_list
    
    def __clean_save_dir(self, model_save_dir: DirectoryPath):
        file_list = os.listdir(model_save_dir)
        ckpt_file_list = [model_save_dir / file for file in file_list if re.match(r"^.*\.ckpt$", file) is not None]
        for ckpt in ckpt_file_list:
            os.remove(ckpt)
            
    def __init_model(self, model_args: DL_ModelArgs, feature_num: int, cate_feature_dict: Dict[int, int]):
        if self._model == MODEL.DNN:
            model = DNN(
                model_args,
                feature_num=feature_num
            )
        elif self._model == MODEL.EBDDNN:
            model = EBDDNN(
                model_args,             # type: ignore
                cate_feature_dict,
                feature_num
            )
        elif self._model == MODEL.FORK:
            model = FORK(
                model_args,             # type: ignore
                cate_feature_dict,
                feature_num
            )
        elif self._model == MODEL.TEST_MODEL:
            model = TEST(
                model_args,             # type: ignore
                cate_feature_dict,
                feature_num
            )
        else:
            raise Exception("model not supported")
        return model