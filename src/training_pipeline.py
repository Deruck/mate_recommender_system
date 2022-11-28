from utils import BaseArguments, PathArgs, LoggerManager
from models import MODEL
from data_module import ZScoreDataModule
from models import model_cli_factory
import datetime

class TrainingPipelineArgs(BaseArguments):
    
    path_args: PathArgs
    
    _model: str
    @property
    def model(self) -> MODEL:
        return MODEL(self._model)
    
    task_name: str
    
    def _add_args(self, parser) -> None:
        parser.add_argument("--model", type=str, required=True, dest="_model", choices=[item.value for item in MODEL])
    
    def _after_parse(self) -> None:
        self.task_name = f"{self.model.value}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        
class TrainingPipeline:
    
    @staticmethod
    def main():
        args = TrainingPipelineArgs().parse_args()
        LoggerManager.set_logger(args.path_args.log_dir / f"{args.task_name}.log")
        data_module = ZScoreDataModule(
            args.path_args.train_dates_csv_file,
            args.path_args.val_dates_csv_file,
            args.path_args.test_dates_csv_file,
            args.path_args.users_csv_file
        )
        model_cli, model_args = model_cli_factory(args.model)
        model_cli.train(model_args, data_module, args.path_args.model_save_dir)
        data_module.dump_data_encoder(args.path_args.data_encoder_file)
        
        
if __name__ == "__main__":
    TrainingPipeline.main()