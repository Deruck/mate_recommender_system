from typing import List

from model_cli_base import BaseModelArgs

class DL_ModelArgs(BaseModelArgs):
    learning_rate = 1e-5
    epochs = 60
    batch_size = 128
    
class SK_ModelArgs(BaseModelArgs):
    ...
    

class DNNModelArgs(DL_ModelArgs):
    learning_rate = 1e-5
    epochs = 60
    batch_size = 128

class EBDDNN_ModelArgs(DL_ModelArgs):
    learning_rate = 1e-5
    epochs = 60
    batch_size = 128
    embedding_size = 8
    
class FORK_ModelArgs(DL_ModelArgs):
    learning_rate = 1e-5
    epochs = 50
    batch_size = 512
    embedding_size = 8