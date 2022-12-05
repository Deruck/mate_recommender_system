import pickle

from utils import BaseArguments, PathArgs
from models import MODEL
from data_module import ZScoreDataModule, BaseDataEncoder, DataReader
from models import model_cli_factory

class InferencePipelineArgs(BaseArguments):
    
    path_args: PathArgs
    
    _model: str
    @property
    def model(self) -> MODEL:
        return MODEL(self._model)
    
    def _add_args(self, parser) -> None:
        parser.add_argument("--model", type=str, required=True, dest="_model", choices=[item.value for item in MODEL])
        
class InferencePipeline:
    
    @staticmethod
    def main():
        args = InferencePipelineArgs().parse_args()
        with open(args.path_args.data_encoder_file, "rb") as f:
            data_encoder: BaseDataEncoder = pickle.load(f)
        user_dict = DataReader.load_user_dict(args.path_args.users_csv_file)
        user_dict = data_encoder.transform(user_dict)
        inf_date_list = DataReader.load_date_list(args.path_args.unlabled_dates_csv_file)
        model, model_args = model_cli_factory(args.model)
        model.load_model(model_args, args.path_args.model_save_dir)
        res = model.inference(inf_date_list, user_dict)
        print(res)
        
        
if __name__ == "__main__":
    InferencePipeline.main()