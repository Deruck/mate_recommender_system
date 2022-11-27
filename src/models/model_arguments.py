from model_cli_base import BaseModelArgs

class DL_ModelArgs(BaseModelArgs):
    
    learning_rate = 1e-5
    epochs = 10
    warmup_epochs = 10
    batch_size = 32
    
class SK_ModelArgs(BaseModelArgs):
    ...