from pydantic import BaseModel
from typing import Literal, Dict, List, Union, Optional, TypedDict
import enum as Enum
from torch import Tensor

#############################################################################################
## 数据加载
#############################################################################################

UserId = int

class User(BaseModel):
    """个人信息"""
    gender: int	
    age: float	
    race: int	
    imprace: float	
    imprelig: float	
    field_cd: int	
    goal: int	
    date: float	
    go_out: float	
    career_c: int	
    sports: float	
    tvsports: float	
    exercise: float	
    dining: float	
    museums: float	
    art: float	
    hiking: float	
    gaming: float	
    clubbing: float	
    reading: float	
    tv: float	
    theater: float	
    movies: float	
    concerts: float	
    music: float
    shopping: float
    yoga: float	
    exphappy: float	
    attr1_1: float	
    sinc1_1: float	
    intel1_1: float	
    fun1_1: float	
    amb1_1: float	
    shar1_1: float	
    attr2_1: float	
    sinc2_1: float	
    intel2_1: float	
    fun2_1: float	
    amb2_1: float	
    shar2_1: float	
    attr3_1: float	
    sinc3_1: float	
    fun3_1: float	
    intel3_1: float	
    amb3_1: float	
    attr5_1: float	
    sinc5_1: float	
    intel5_1: float	
    fun5_1: float	
    amb5_1: float
    
class Date(BaseModel):
    """约会信息"""
    iid: UserId
    pid: UserId
    dec: Optional[float]

UserDict = Dict[UserId, User]
DateList = List[Date]

CATEGORICAL_VARIABLES = (
    "gender",
    "race",
    "field_cd",
    "goal",
    "career_c"
)

NUMERICAL_VARIABLES = tuple(set(User.__fields__.keys()).difference(set(CATEGORICAL_VARIABLES)))

CATEGORICAL_UNIQUE = {
    "gender": 2,
    "race": 6,
    "field_cd": 18,
    "goal": 6,
    "career_c": 17
}

#############################################################################################
## 数据编码
#############################################################################################
    
class UserD(TypedDict):
    """个人信息"""
    gender: int	
    age: float	
    race: int	
    imprace: float	
    imprelig: float	
    field_cd: int	
    goal: int	
    date: float	
    go_out: float	
    career_c: int	
    sports: float	
    tvsports: float	
    exercise: float	
    dining: float	
    museums: float	
    art: float	
    hiking: float	
    gaming: float	
    clubbing: float	
    reading: float	
    tv: float	
    theater: float	
    movies: float	
    concerts: float	
    music: float
    shopping: float
    yoga: float	
    exphappy: float	
    attr1_1: float	
    sinc1_1: float	
    intel1_1: float	
    fun1_1: float	
    amb1_1: float	
    shar1_1: float	
    attr2_1: float	
    sinc2_1: float	
    intel2_1: float	
    fun2_1: float	
    amb2_1: float	
    shar2_1: float	
    attr3_1: float	
    sinc3_1: float	
    fun3_1: float	
    intel3_1: float	
    amb3_1: float	
    attr5_1: float	
    sinc5_1: float	
    intel5_1: float	
    fun5_1: float	
    amb5_1: float
    
#############################################################################################
## 模型
#############################################################################################

class UserTensor(TypedDict):
    """个人信息"""
    gender: Tensor	
    age: Tensor	
    race: Tensor	
    imprace: Tensor	
    imprelig: Tensor	
    field_cd: Tensor	
    goal: Tensor	
    date: Tensor	
    go_out: Tensor	
    career_c: Tensor	
    sports: Tensor	
    tvsports: Tensor	
    exercise: Tensor	
    dining: Tensor	
    museums: Tensor	
    art: Tensor	
    hiking: Tensor	
    gaming: Tensor	
    clubbing: Tensor	
    reading: Tensor	
    tv: Tensor	
    theater: Tensor	
    movies: Tensor	
    concerts: Tensor	
    music: Tensor
    shopping: Tensor
    yoga: Tensor	
    exphappy: Tensor	
    attr1_1: Tensor	
    sinc1_1: Tensor	
    intel1_1: Tensor	
    fun1_1: Tensor	
    amb1_1: Tensor	
    shar1_1: Tensor	
    attr2_1: Tensor	
    sinc2_1: Tensor	
    intel2_1: Tensor	
    fun2_1: Tensor	
    amb2_1: Tensor	
    shar2_1: Tensor	
    attr3_1: Tensor	
    sinc3_1: Tensor	
    fun3_1: Tensor	
    intel3_1: Tensor	
    amb3_1: Tensor	
    attr5_1: Tensor	
    sinc5_1: Tensor	
    intel5_1: Tensor	
    fun5_1: Tensor	
    amb5_1: Tensor

class TrainBatch(TypedDict):
    subject: Tensor
    object: Tensor
    dec: Tensor
    
class Batch(TypedDict):
    subject: Tensor
    object: Tensor

class TrainModelInput(TypedDict):
    subject: Tensor
    object: Tensor
    dec: float

class ModelOutput(TypedDict):
    probs: Tensor
    
class InfModelInput(TypedDict):
    subject: Tensor
    object: Tensor

class InfModelOutput(TypedDict):
    predicts: List[float]
    probs: List[float]
