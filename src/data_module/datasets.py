from torch.utils.data import Dataset
from typing import Dict
from torch import Tensor
import torch

from entities import UserDict, DateList, TrainModelInput, InfModelInput, UserId, UserD, User

class BaseDataset(Dataset):
    def __init__(self, date_list: DateList, user_dict: UserDict) -> None: 
        self._date_list = date_list
        self._user_dict: Dict[UserId, Tensor] = {user_id: self._user_to_tensor(user) for user_id, user in user_dict.items()}
    
    def __getitem__(self, index: int):
        raise NotImplementedError()
    
    def __len__(self):
        return len(self._date_list)
    
    def _user_to_tensor(self, user: User) -> Tensor:
        return torch.tensor([value for value in user.dict().values()]).to(torch.float32)


class TrainDataset(BaseDataset):
    
    def __getitem__(self, index: int) -> TrainModelInput:
        date = self._date_list[index]
        assert date.dec is not None
        return TrainModelInput(
            subject=self._user_dict[date.iid], 
            object=self._user_dict[date.pid],
            dec=date.dec
        )
        
        
class InfDataset(BaseDataset):
    
    def __getitem__(self, index: int) -> InfModelInput:
        date = self._date_list[index]
        return InfModelInput(
            subject=self._user_dict[date.iid], 
            object=self._user_dict[date.pid],
        )
