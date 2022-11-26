from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from pydantic import FilePath
from abc import ABCMeta, abstractproperty
from typing import Type, Set, Tuple
import pickle

from data_encoder import BaseDataEncoder, ZScoreDataEncoder
from data_reader import DataReader
from datasets import TrainDataset, InfDataset
from entities import UserDict, DateList, UserDict


class BaseDataModule(LightningDataModule):
    
    def __init__(
        self, 
        train_date_path: FilePath,
        val_date_path: FilePath,
        test_date_path: FilePath,
        user_info_path: FilePath,
    ) -> None:
        self.train_date_list = DataReader.load_date_list(train_date_path)
        self.val_date_list = DataReader.load_date_list(val_date_path)
        self.test_date_list = DataReader.load_date_list(test_date_path)
        self.user_info_dict = DataReader.load_user_dict(user_info_path)
        
        self.train_user_info = self._get_user_dict_for_date_list(self.train_date_list)
        self.val_user_info = self._get_user_dict_for_date_list(self.val_date_list)
        self.test_user_info = self._get_user_dict_for_date_list(self.test_date_list)
        
        self._data_encoder = self._data_encoder_cls()
        self.train_encoded_user_info = self._data_encoder.fit_transform(self.train_user_info)
        self.val_encoded_user_info = self._data_encoder.transform(self.val_user_info)
        self.test_encoded_user_info = self._data_encoder.transform(self.test_user_info)
    
    def train_dataloader(self, batch_size: int = 32, num_workers: int = 6) -> DataLoader:
        dataset = TrainDataset(self.train_date_list, self.train_encoded_user_info)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    def val_dataloader(self, batch_size: int = 32, num_workers: int = 6) -> DataLoader:
        dataset = TrainDataset(self.val_date_list, self.val_encoded_user_info)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    def test_dataloader(self, batch_size: int = 32, num_workers: int = 6) -> DataLoader:
        dataset = TrainDataset(self.test_date_list, self.test_encoded_user_info)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    def dump_data_encoder(self, dump_path: FilePath) -> None:
        with open(dump_path, "wb") as f:
            pickle.dump(self._data_encoder, f)
            
    @property
    def train_sample_size(self) -> int:
        return len(self.train_date_list)
    
    @property
    def val_sample_size(self) -> int:
        return len(self.val_date_list)
    
    @property
    def test_sample_size(self) -> int:
        return len(self.test_date_list)
    
    def _get_user_dict_for_date_list(self, date_list: DateList) -> UserDict:
        user_id_set: Set[int] = set()
        for date in date_list:
            user_id_set.add(date.iid)
            user_id_set.add(date.pid)
        return {user_id: self.user_info_dict[user_id] for user_id in user_id_set}
    
    @abstractproperty
    def _data_encoder_cls(self) -> Type[BaseDataEncoder]: ...
    
class ZScoreDataModule(BaseDataModule):
    
    @property
    def _data_encoder_cls(self) -> Type[BaseDataEncoder]: 
        return ZScoreDataEncoder