import pickle
from typing import Dict, List, Union, Sequence, NamedTuple, Callable, Tuple, TypedDict
from itertools import product
from pydantic import FilePath, DirectoryPath

from utils import BaseArguments, PathArgs
from models import MODEL
from data_module import ZScoreDataModule, BaseDataEncoder, DataReader
from models import model_cli_factory
from data_module.entities import Date, UserDict, User, UserId
import json

class RecommenderArgs(BaseArguments):
    
    path_args: PathArgs
    
    _model: str
    @property
    def model(self) -> MODEL:
        return MODEL(self._model)
    
    def _add_args(self, parser) -> None:
        parser.add_argument("--model", type=str, required=True, dest="_model", choices=[item.value for item in MODEL])
        
class Recommend(TypedDict):
    like: Dict[UserId, float]
    being_liked: Dict[UserId, float]
    match_score: Dict[UserId, float]

class Recommender:
    
    def __init__(self, model: MODEL, data_encoder_file: FilePath, model_save_dir: DirectoryPath):
        with open(data_encoder_file, "rb") as f:
            self._data_encoder: BaseDataEncoder = pickle.load(f)
        self._user_pool: UserDict = {}
        self._next_user_id: int = 0
        self._model, model_args = model_cli_factory(model)
        self._model.load_model(model_args, model_save_dir)
    
    def add_to_user_pool(self, user_dict: UserDict):
        self._user_pool.update(user_dict)
        
    def recommend(self, user_dict: UserDict, top_n: int = 5) -> Dict[UserId, Recommend]:
        date_list = self.__generate_input_date_list(user_dict)
        all_user_info = {}
        all_user_info.update(self._user_pool)
        all_user_info.update(user_dict)
        all_user_info = self._data_encoder.transform(all_user_info)
        output_date_list = self._model.inference(date_list, all_user_info)
        score_dict = self.__parse_output_date_list(output_date_list, top_n=top_n)
        score_dict = {key: val for key, val in score_dict.items() if key in user_dict.keys()}
        return score_dict
    
    def __generate_input_date_list(self, user_dict: UserDict) -> List[Date]:
        date_list: List[Date] = []
        for user_id, candicate_user_id in product(user_dict.keys(), self._user_pool.keys()):
            if user_dict[user_id].gender == self._user_pool[candicate_user_id].gender:
                continue
            sub_date = Date(iid=user_id, pid=candicate_user_id, dec=None)
            # if sub_date not in date_list:
            date_list.append(sub_date)
            obj_date = Date(iid=candicate_user_id, pid=user_id, dec=None)
            # if obj_date not in date_list:
            date_list.append(obj_date)
        return date_list
    
    def __parse_output_date_list(self, date_list: List[Date], top_n: int) -> Dict[UserId, Recommend]:
        score_dict: Dict[UserId, Dict[UserId, float]] = {}
        for date in date_list:
            if date.iid not in score_dict.keys():
                score_dict[date.iid] = {}
            score_dict[date.iid][date.pid] = date.dec # type: ignore
        res: Dict[UserId, Recommend] = {}
        for iid in score_dict.keys():
            like = {pid: round(100 * score_dict[iid][pid], 2) for pid in score_dict[iid].keys()}
            being_liked = {pid: round(100 * score_dict[pid][iid], 2) for pid in score_dict[iid].keys()}
            match_score = {pid: round(100 * score_dict[iid][pid] * score_dict[pid][iid], 2) for pid in score_dict[iid].keys()}
            res[iid] = Recommend(
                like=self.__sort_score_dict(like, top_n=top_n), 
                being_liked=self.__sort_score_dict(being_liked, top_n=top_n), 
                match_score=self.__sort_score_dict(match_score, top_n=top_n)
            )
        return res
    
    def __sort_score_dict(self, score_dict: Dict[UserId, float], top_n: int) -> Dict[UserId, float]:
        sort_key: Callable[[Tuple[UserId, float]], float] = lambda recommend: recommend[1]
        return dict(sorted(score_dict.items(), key=sort_key, reverse=True)[:top_n])
    


if __name__ == "__main__":
    args = RecommenderArgs().parse_args()
    recommender = Recommender(args.model, args.path_args.data_encoder_file, args.path_args.model_save_dir)
    user_dict = DataReader.load_user_dict(args.path_args.unlabled_dates_csv_file)
    # user_pool = DataReader.load_user_dict(args.path_args.users_csv_file)
    # recommender.add_to_user_pool(user_pool)
    recommender.add_to_user_pool(user_dict)
    recommendation = recommender.recommend(user_dict, 30)
    json.dump(recommendation, open(args.path_args.out_dir / "recommendations.json", "w", encoding="utf-8"), indent=2, sort_keys=False)