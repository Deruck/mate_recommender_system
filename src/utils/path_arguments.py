from pydantic import DirectoryPath, FilePath
from pathlib import Path

from utils.base_arguments import BaseArguments

class PathArgs(BaseArguments):

    _data_dir: str
    @property
    def data_dir(self) -> DirectoryPath:
        return Path(self._data_dir)
    
    _log_dir: str
    @property
    def log_dir(self) -> DirectoryPath:
        return Path(self._log_dir)
    
    _out_dir: str
    @property
    def out_dir(self) -> DirectoryPath:
        return Path(self._out_dir)
    
    def _add_args(self, parser) -> None:
        parser.add_argument("--log-dir", dest="_log_dir", help="日志目录", default="./logs")
        parser.add_argument("--data-dir", dest="_data_dir", help="数据目录", default="./data")
        parser.add_argument("--out_dir", dest="_out_dir", help="输出目录", default="./out")
        
    @property
    def raw_data_dir(self) -> DirectoryPath:
        return self.data_dir / "raw"
    
    @property
    def pre_processed_data_dir(self) -> DirectoryPath:
        return self.data_dir / "pre_processed"  
    
    @property
    def processed_data_dir(self) -> DirectoryPath:
        return self.data_dir / "processed"
    
    @property
    def train_dates_csv_file(self) -> FilePath:
        return self.pre_processed_data_dir / "dates_train.csv"
    
    @property
    def val_dates_csv_file(self) -> FilePath:
        return self.pre_processed_data_dir / "dates_val.csv"
    
    @property
    def test_dates_csv_file(self) -> FilePath:
        return self.pre_processed_data_dir / "dates_test.csv"
    
    @property
    def unlabled_dates_csv_file(self) -> FilePath:
        return self.pre_processed_data_dir / "dates.csv"
    
    @property
    def users_csv_file(self) -> FilePath:
        return self.pre_processed_data_dir / "users.csv"
    
    @property
    def data_encoder_file(self) -> FilePath:
        return self.out_dir / "data_encoder.bin"
    
    @property
    def model_save_dir(self) -> DirectoryPath:
        return self.out_dir / "models"

    def _arg_name(self):
        return "通用路径参数"
    
    def _after_parse(self) -> None:
        for dir in (
            self.data_dir, 
            self.log_dir, 
            self.out_dir,
            self.processed_data_dir,
            self.model_save_dir
        ):
            if not dir.exists():
                dir.mkdir()
