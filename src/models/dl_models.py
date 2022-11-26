from pytorch_lightning import LightningModule
from transformers.optimization import AdamW
from abc import ABCMeta, abstractmethod
from torch import nn, Tensor
from typing import Dict, List, Callable, Type
import torch
from numpy import vectorize
from enum import Enum

from utils import warmup_linear_decay_scheduler_factory, LoggerManager
from data_module.entities import TrainBatch, ModelOutput, Batch
from model_arguments import DL_ModelArgs
from consts import MODEL

logger = LoggerManager.get_logger()

#############################################################################################
## Base
#############################################################################################

class BaseModel(LightningModule):
    
    #############################################################################################
    ##################################### configs #############################################
    
    def __init__(
        self,
        train_args: DL_ModelArgs,
        train_sample_size: int
    ):
        self._train_args = train_args
        self._train_sample_size = train_sample_size
        self._ce_loss = nn.BCELoss()
        self._probs_to_predicts: Callable[[Tensor], Tensor] = vectorize(lambda x: 0 if x < 0.5 else 1)
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr = self._train_args.learning_rate,
            eps=1e-8,
            correct_bias=True
        )
        scheduler = warmup_linear_decay_scheduler_factory(
            optimizer=optimizer,
            warm_up_epoch=self._train_args.warmup_epochs,
            decay_epoch=self._train_args.epochs - 1,
            epoch=self._train_args.epochs,
            train_data_length=self._train_sample_size,
            batch_size=self._train_args.batch_size,
            min_lr=1e-8
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config
        }
    
    #############################################################################################
    ##################################### training #############################################
        
    def training_step(self, batch: TrainBatch, batch_idx: int):
        probs = self.forward(Batch(subject=batch["subject"], object=batch["object"]))["probs"]
        loss = self._ce_loss(probs, batch["dec"])
        lr = self.lr_schedulers().get_last_lr()[-1] # type: ignore
        metrics = {
            "lr": lr,
            "train_loss": loss,
        }
        self.log_dict(metrics)
        return loss
    
    #############################################################################################
    ##################################### validation #############################################
    
    def on_validation_epoch_start(self) -> None:
        logger.info("validating...")
    
    def validation_step(self, batch: TrainBatch, batch_idx: int):
        probs = self.forward(Batch(subject=batch["subject"], object=batch["object"]))["probs"]
        loss: Tensor = self._ce_loss(probs, batch["dec"])
        predicts = self._probs_to_predicts(probs)
        acc_num = torch.eq(predicts, probs)
        batch_size = len(batch["dec"])
        return {"loss": loss, "acc_num": acc_num, "batch_size": batch_size} 
    
    def validation_epoch_end(self, validation_step_outputs: List[Dict[str, Tensor]]) -> None:
        """聚合所有验证结果并打印"""
        loss_list = torch.stack([step["loss"] for step in validation_step_outputs])
        acc_num_list = torch.stack([step["acc_num"] for step in validation_step_outputs])
        batch_size_list = torch.stack([step["batch_size"] for step in validation_step_outputs])
        dev_loss = loss_list.mean()
        dev_acc = acc_num_list.sum() / batch_size_list.sum()
        metrics = {
            "dev_loss": dev_loss,
            "dev_acc": dev_acc
        }
        self.log_dict(metrics, prog_bar=False, logger=True, rank_zero_only=True)
        logger.info(f"validation - loss: {dev_loss:.4f} acc: {dev_acc:.4f} epoch: {self.current_epoch}")
    
    #############################################################################################
    ##################################### inference #############################################
    
    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> List[int]:
        probs = self.forward(Batch(subject=batch["subject"], object=batch["object"]))["probs"]
        prodicts: List[int] = self._probs_to_predicts(probs).int().tolist()
        return prodicts
    
    def on_predict_epoch_end(self, results: List[List[int]]) -> List[int]:
        """聚合每个batch的预测结果"""
        predicts = sum(results, [])
        return predicts
    
    #############################################################################################
    ##################################### abstract #############################################
        
    @abstractmethod
    def forward(self, batch: Batch) -> ModelOutput: ...

#############################################################################################
## Subs
#############################################################################################

class DNN(BaseModel):
    
    def forward(self, batch: Batch):
        return torch.zeros_like(batch["subject"]["age"])

#############################################################################################
## api
#############################################################################################

def get_dl_model(model: MODEL) -> Type[BaseModel]:
    if model == MODEL.DNN:
        return DNN
    else:
        raise Exception("model not supported")




