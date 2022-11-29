from pydantic import FilePath
from typing import Tuple, List, Optional, Dict
from numpy.typing import NDArray
import numpy as np
from copy import deepcopy
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from json import dumps
import pickle
import shutil

from data_module import BaseDataModule
from data_module.entities import DateList, UserDict
from utils import LoggerManager, evaluate_model

from model_cli_base import BaseModelCli
from consts import MODEL
from sklearn_models import get_sklearn_model, ClassifierProto
from model_arguments import DL_ModelArgs

logger = LoggerManager.get_logger()

class SKLModelCli(BaseModelCli):
    
    def __init__(self, model: MODEL):
        self._model_name: str = model.value
        self._model = get_sklearn_model(model)
        
    def train(self, model_args: DL_ModelArgs, data_module: BaseDataModule, model_save_dir):
        logger.info(f"training model: {self._model_name}")
        date_list = data_module.train_date_list + data_module.val_date_list
        user_dict = deepcopy(data_module.train_encoded_user_info)
        user_dict.update(data_module.val_encoded_user_info)
        test_date_list = data_module.test_date_list
        test_user_dict = data_module.test_encoded_user_info
        cate_feature_dict = data_module.cate_feature_dict
        x, y = self.__get_model_input(date_list, user_dict, cate_feature_dict)
        x_test, y_test = self.__get_model_input(test_date_list, test_user_dict, cate_feature_dict)
        gs = GridSearchCV(
            estimator=self._model.model_cls(**self._model.other_params), # type: ignore
            param_grid=self._model.gs_params,
            scoring="f1",
            n_jobs=-1,
            cv=5,
            verbose=10,
        )
        gs.fit(x, y)
        best_model: ClassifierProto = gs.best_estimator_ # type: ignore
        best_model.fit(x, y)
        logger.info(
            f"""best model params
            {dumps(gs.best_params_, indent=2)}
            """
        )
        probs = best_model.predict_proba(x_test)[:, 1].tolist()
        evaluate_model(probs, y_test.tolist())
        model_save_dir = model_save_dir / self._model_name
        if not model_save_dir.exists():
            model_save_dir.mkdir()
        save_path = model_save_dir /f"{self._model_name}.bin"
        logger.info(f"save model at {save_path}")
        with open(save_path, "wb") as f: 
            pickle.dump(best_model, f)
        log_file = LoggerManager.get_log_file()
        shutil.copy(log_file, model_save_dir / "train_log.log")
        
    
    def load_model(self, model_save_path: FilePath):
        return super().load_model(model_save_path)
    
    def inference(self, date_list: DateList, user_dict: UserDict) -> DateList:
        return super().inference(date_list, user_dict)
    
    def __get_model_input(self, date_list: DateList, user_dict: UserDict, cate_feature_dict: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        sub_input_list: List[List[float]] = []
        obj_input_list: List[List[float]] = []
        label_list: List[Optional[int]] = []
        for date in date_list:
            sub_value_array: NDArray[np.float32] = np.array(self.__id_to_value_list(date.iid, user_dict))
            obj_value_array: NDArray[np.float32] = np.array(self.__id_to_value_list(date.pid, user_dict))
            sub_input_list.append(sub_value_array.tolist())
            obj_input_list.append(obj_value_array.tolist())
            label_list.append(date.dec)
        sub_input_array = self.__onehot_encode(np.array(sub_input_list), cate_feature_dict)
        obj_input_array = self.__onehot_encode(np.array(obj_input_list), cate_feature_dict)
        return sub_input_array - obj_input_array, np.array(label_list)
        
    def __id_to_value_list(self, id: int, user_dict: UserDict) -> List[float]:
        return [value for value in user_dict[id].dict().values()]
    
    def __onehot_encode(self, x: np.ndarray, cate_feature_dict: Dict[int, int]) -> np.ndarray:
        cate_fea_idx = list(cate_feature_dict.keys())
        num_fea_idx = list(set(range(x.shape[1])).difference(set(cate_fea_idx)))
        num_features = x[:, num_fea_idx] # type: ignore
        cate_features = x[:, cate_fea_idx] # type: ignore
        encoder = OneHotEncoder(categories=[range(unique_num) for unique_num in cate_feature_dict.values()], sparse=False) # type: ignore
        cate_features_encoded = encoder.fit_transform(cate_features)
        return np.concatenate([num_features, cate_features_encoded], axis=1)