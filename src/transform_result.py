import os
import sys
sys.path.append(os.path.dirname(__file__))

from pathlib import Path
from pydantic import FilePath, DirectoryPath
from typing import List, NamedTuple, Dict

from recommender import Recommend
from utils import BaseArguments
from utils.json import load_json
from data_module.entities import UserId

class Args(BaseArguments):
    result_path: FilePath
    save_dir: DirectoryPath
    
    def _add_args(self, parser) -> None:
        parser.add_argument("--result_path", type=Path)
        parser.add_argument("--save_dir", type=Path)
        
    def _after_parse(self) -> None:
        self.save_dir.mkdir()
        
class CsvLine(NamedTuple):
    top_like: str
    like_prob: str
    top_being_like: str
    being_like_prob: str
    top_match: str
    match_prob: str

class RecommendResultTransformer:        
    
    @classmethod
    def transform_result(cls, result_path: FilePath, save_dir: DirectoryPath, top_n: int = 8) -> None:
        recommend_dict = load_json(result_path, Dict[UserId, Recommend])
        for user_id, recommend in recommend_dict.items():
            csv_lines = cls.__transform_recommend(recommend, top_n)
            cls.__dump_csv(csv_lines, f"result_{user_id}.csv", save_dir)
        
    @classmethod
    def __transform_recommend(cls, recommend: Recommend, top_n: int = 8) -> List[CsvLine]:
        res: List[CsvLine] = []
        for like, being_liked, match in zip(recommend["like"].items(), recommend["being_liked"].items(), recommend["match_score"].items()):
            res.append(CsvLine(
                top_like=str(like[0]),
                like_prob=str(like[1]),
                top_being_like=str(being_liked[0]),
                being_like_prob=str(being_liked[1]),
                top_match=str(match[0]),
                match_prob=str(match[1])
            ))
        return res
        
    @classmethod
    def __dump_csv(cls, lines: List[CsvLine], file_name: str, save_dir: DirectoryPath) -> None:
        headers = ["top_like", "like_prob", "top_being_like", "being_like_prob", "top_match", "match_prob"]
        with open(save_dir / file_name, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for line in lines:
                f.write(",".join(list(line)) + "\n")
        

if __name__ == "__main__":
    args = Args().parse_args()
    RecommendResultTransformer.transform_result(args.result_path, args.save_dir)

