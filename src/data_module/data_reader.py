from pydantic import FilePath
from typing import List, Dict
import pandas as pd

from entities import User, Date, UserDict, DateList



class DataReader:
    
    @classmethod
    def load_user_dict(cls, user_csv_file_path: FilePath) -> UserDict:
        with open(user_csv_file_path, "r", encoding="utf-8") as f:
            df = pd.read_csv(f)
        user_dict: UserDict = {
            int(dict_["iid"]): User(**dict_)
            for dict_ in df.T.to_dict().values()
        }
        return user_dict
    
    @classmethod
    def load_date_list(cls, date_csv_file_path: FilePath) -> DateList:
        with open(date_csv_file_path, "r", encoding="utf-8") as f:
            df = pd.read_csv(f)
        date_list = [
            Date(**dict_)
            for dict_ in df.T.to_dict().values()
        ]
        return date_list
        

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from utils.path_arguments import PathArgs
    
    args = PathArgs().parse_args()
    
    user_dict = DataReader.load_user_dict(args.users_csv_file)
    date_list = DataReader.load_date_list(args.dates_csv_file)
    print(list(user_dict.items())[:5])
    print(date_list[:5])