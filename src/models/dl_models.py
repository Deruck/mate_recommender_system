from pytorch_lightning import LightningModule
from transformers.optimization import AdamW
from abc import ABCMeta, abstractmethod
from torch import nn, Tensor
from typing import Dict, List, Callable, Type
import torch
from enum import Enum
from torch import sigmoid

from utils import warmup_linear_decay_scheduler_factory, LoggerManager
from data_module.entities import TrainBatch, ModelOutput, Batch
from model_arguments import DL_ModelArgs, EBDDNN_ModelArgs, FORK_ModelArgs
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
        model_args: DL_ModelArgs
    ):
        super().__init__()
        self._model_args = model_args
        self._ce_loss = nn.BCELoss()
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr = self._model_args.learning_rate,
            eps=1e-8,
            correct_bias=True
        )
        return {
            "optimizer": optimizer,
        }
    
    #############################################################################################
    ##################################### training #############################################
        
    def training_step(self, batch: TrainBatch, batch_idx: int):
        probs = self.forward(Batch(subject=batch["subject"], object=batch["object"]))["probs"]
        dec = batch["dec"].to(torch.float32)
        loss = self._ce_loss(probs, dec)
        metrics = {
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
        dec = batch["dec"].to(torch.float32)
        loss: Tensor = self._ce_loss(probs, dec)
        predicts = (probs > 0.5).to(torch.float32)
        acc_num = torch.eq(predicts, dec).sum()
        batch_size = torch.tensor(dec.shape[0])
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
        predicts: List[int] = (probs > 0.5).int().tolist()
        return predicts
    
    def on_predict_epoch_end(self, results):
        """聚合每个batch的预测结果"""
        results[0] = sum(results[0], [])
    
    #############################################################################################
    ##################################### abstract #############################################
        
    @abstractmethod
    def forward(self, batch: Batch) -> ModelOutput: ...

#############################################################################################
## Subs
#############################################################################################

class DNN(BaseModel):
    def __init__(self, model_args: DL_ModelArgs, feature_num: int):
        super().__init__(model_args)
        self._dnn = nn.Sequential(
            nn.Linear(2 * feature_num, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
    
    def forward(self, batch: Batch) -> ModelOutput:
        input = torch.concat([batch["subject"], batch["object"]], 1)
        logits  = self._dnn.forward(input)
        probs = sigmoid(logits)
        probs = probs.reshape(-1)
        return ModelOutput(probs = probs)


class EBDDNN(BaseModel):
    
    def __init__(self, model_args: EBDDNN_ModelArgs, cate_feature_dict: Dict[int, int], feature_num: int):
        super().__init__(model_args)
        self._cate_feature_dict = cate_feature_dict
        self._embeddings = nn.ModuleList([nn.Embedding(dim, model_args.embedding_size) for dim in self._cate_feature_dict.values()])
        self._feature_num = feature_num
        self._num_feature_num = feature_num - len(cate_feature_dict)
        self._cate_embedding_num = len(cate_feature_dict) * model_args.embedding_size
        self._cate_fea_idx = list(self._cate_feature_dict.keys())
        self._num_fea_idx = list(set(range(self._feature_num)).difference(set(self._cate_fea_idx)))
        self._dnn = nn.Sequential(
            nn.Linear(2 * (self._num_feature_num + self._cate_embedding_num), 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        
    
    def forward(self, batch: Batch) -> ModelOutput:
        sub_cate_features, sub_num_features = self.__splite_cate_num(batch["subject"])
        obj_cate_features, obj_num_features = self.__splite_cate_num(batch["object"])
        sub_cate_embeddings = torch.concat([embedding(sub_cate_features[:, idx].int()) for idx, embedding in enumerate(self._embeddings)], 1)
        obj_cate_embeddings = torch.concat([embedding(obj_cate_features[:, idx].int()) for idx, embedding in enumerate(self._embeddings)], 1)
        sub_input = torch.concat([sub_num_features, sub_cate_embeddings], 1)
        obj_input = torch.concat([obj_num_features, obj_cate_embeddings], 1)
        input = torch.concat([sub_input, obj_input], 1)
        logits  = self._dnn.forward(input)
        probs = sigmoid(logits)
        probs = probs.reshape(-1)
        return ModelOutput(probs = probs)
    
    
    def __splite_cate_num(self, user_tensor: Tensor):
        return user_tensor[:, self._cate_fea_idx], user_tensor[:, self._num_fea_idx]
        
class FORK(BaseModel):
    
    def __init__(self, model_args: FORK_ModelArgs, cate_feature_dict: Dict[int, int], feature_num: int):
        super().__init__(model_args)
        self._cate_feature_dict = cate_feature_dict
        self._embeddings = nn.ModuleList([nn.Embedding(dim, model_args.embedding_size) for dim in self._cate_feature_dict.values()])
        self._feature_num = feature_num
        self._num_feature_num = feature_num - len(cate_feature_dict)
        self._cate_embedding_num = len(cate_feature_dict) * model_args.embedding_size
        self._cate_fea_idx = list(self._cate_feature_dict.keys())
        self._num_fea_idx = list(set(range(self._feature_num)).difference(set(self._cate_fea_idx)))
        self._dnn = nn.Sequential(
            nn.Linear((self._num_feature_num + self._cate_embedding_num), 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
        )
        self._dnn_sub = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
        )
        self._dnn_obj = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
        )
        
    def forward(self, batch: Batch) -> ModelOutput:
        sub_cate_features, sub_num_features = self.__splite_cate_num(batch["subject"])
        obj_cate_features, obj_num_features = self.__splite_cate_num(batch["object"])
        sub_cate_embeddings = torch.concat([embedding(sub_cate_features[:, idx].int()) for idx, embedding in enumerate(self._embeddings)], 1)
        obj_cate_embeddings = torch.concat([embedding(obj_cate_features[:, idx].int()) for idx, embedding in enumerate(self._embeddings)], 1)
        sub_input = torch.concat([sub_num_features, sub_cate_embeddings], 1)
        obj_input = torch.concat([obj_num_features, obj_cate_embeddings], 1)
        sub_logits  = self._dnn_sub.forward(self._dnn.forward(sub_input))
        obj_logits  = self._dnn_obj.forward(self._dnn.forward(obj_input))
        logits = torch.mul(sub_logits, obj_logits).sum(dim=1)
        probs = sigmoid(logits)
        return ModelOutput(probs=probs)
    
    
    def __splite_cate_num(self, user_tensor: Tensor):
        return user_tensor[:, self._cate_fea_idx], user_tensor[:, self._num_fea_idx]



#############################################################################################
## api
#############################################################################################

def get_dl_model(model: MODEL) -> Type[BaseModel]:
    if model == MODEL.DNN:
        return DNN
    else:
        raise Exception("model not supported")




