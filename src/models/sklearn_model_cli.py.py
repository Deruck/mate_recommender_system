from pydantic import FilePath
from typing import Tuple, List, Optional
from numpy.typing import NDArray
import numpy as np
from copy import deepcopy
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from data_module import BaseDataModule
from data_module.entities import DateList, UserDict
from utils import LoggerManager

from model_cli_base import BaseModelCli
from consts import MODEL
from sklearn_models import get_sklearn_model, ClassifierProto
from model_arguments import DL_ModelArgs

logger = LoggerManager.get_logger()

class SKLModelCli(BaseModelCli):
    
    def __init__(self, model: MODEL):
        self._model_name = model.value
        self._model = get_sklearn_model(model)
        
    def train(self, model_args: DL_ModelArgs, data_module: BaseDataModule):
        date_list = data_module.train_date_list + data_module.val_date_list
        user_dict = deepcopy(data_module.train_encoded_user_info)
        user_dict.update(data_module.val_encoded_user_info)
        test_date_list = data_module.test_date_list
        test_user_dict = data_module.test_encoded_user_info
        x, y = self.__get_model_input(date_list, user_dict)
        x_test, y_test = self.__get_model_input(test_date_list, test_user_dict)
        gs = GridSearchCV(
            estimator=self._model.model_cls(**self._model.other_params), # type: ignore
            param_grid=self._model.gs_params,
            scoring="f1",
            n_jobs=-1,
            cv=5,
            verbose=1
        )
        gs.fit(x, y)
        best: ClassifierProto = gs.best_estimator_ # type: ignore
        predict = best.predict(x_test)
        acc = accuracy_score(predict, y_test)
        logger.info(f"best {self._model_name} estimator acc: {acc:.4f}")
        return super().train(model_args, data_module)
    
    def load_model(self, model_save_path: FilePath):
        return super().load_model(model_save_path)
    
    def inference(self, date_list: DateList, user_dict: UserDict) -> DateList:
        return super().inference(date_list, user_dict)
    
    def __get_model_input(self, date_list: DateList, user_dict: UserDict) -> Tuple[np.ndarray, np.ndarray]:
        input_list: List[List[float]] = []
        label_list: List[Optional[int]] = []
        for date in date_list:
            sub_value_array: NDArray[np.float32] = np.array(self.__id_to_value_list(date.iid, user_dict))
            obj_value_array: NDArray[np.float32] = np.array(self.__id_to_value_list(date.pid, user_dict))
            input_list.append((sub_value_array - obj_value_array).tolist())
            label_list.append(date.dec)
        return np.array(input_list), np.array(label_list)
        
    def __id_to_value_list(self, id: int, user_dict: UserDict) -> List[float]:
        return [value for value in user_dict[id].dict().values()]