import os

from pathlib import Path

from typing import List, Literal, Optional, Tuple, Union, Dict
from typing_extensions import Annotated

from pydantic import AfterValidator, BaseModel, conlist, Field, ConfigDict

def ml_model_change(value: Union[List[str], str]) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if "all" in value:
        return ["logreg","knn","svm","gaussian","tree","rf","nn","adaboost","nb","qda","xgb"]
    return value

def ml_file_change(value: Union[List[Path], Path]) -> List[Path]:
    if isinstance(value, Path):
        if value.is_dir():
            value = [f for f in value.iterdir() if f.is_file()]
        elif value.is_file():
            value = [value]
        else:
            raise ValueError(f"{value} is neither a directory or a file")
    
    if len(value) == 0:
        raise ValueError(f"Not enough machine learning training datasets")
    
    for path in value:
        if not path.exists():
            raise ValueError(f"{path} machine learning training dataset does not exist")
        if not path.is_file():
            raise ValueError(f"{path} machine learning training dataset is not a file")
        if not os.access(str(path), os.R_OK):
            raise ValueError(f"{path} machine learning training dataset does not have read permissions")
        
    return value 

def check_path(value: Path) -> Path:
    if not value.exists():
        raise ValueError(f"{value} ML train output dir does not exist")
    if not value.is_dir():
        raise ValueError(f"{value} ML train output dir does is not a directory")
    if not os.access(str(value), os.R_OK):
        raise ValueError(f"{value} ML train output dir does not have read permissions")
    if not os.access(str(value), os.W_OK):
        raise ValueError(f"{value} ML train output dir does not have write permissions")
    
    return value

class MLTrainConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    models: Annotated[Union[
    conlist(Literal["logreg","knn","svm","gaussian","tree","rf","nn","adaboost","nb","qda","xgb", "all"], min_length=1),
    Literal["logreg","knn","svm","gaussian","tree","rf","nn","adaboost","nb","qda","xgb", "all"]
], AfterValidator(ml_model_change)]
    datasets: Annotated[Union[List[Path], Path], AfterValidator(ml_file_change)]
    train_split: Optional[float] = Field(default=None)
    output_dir: Annotated[Path, AfterValidator(check_path)]
    logreg: Optional[Dict[str, Union[float, str, int]]] = Field(default=None)
    knn: Optional[Dict[str, Union[float, str, int]]] = Field(default=None)
    gaussian: Optional[Dict[str, Union[float, str, int]]] = Field(default=None)
    tree: Optional[Dict[str, Union[float, str, int]]] = Field(default=None)
    rf: Optional[Dict[str, Union[float, str, int]]] = Field(default=None)
    nn: Optional[Dict[str, Union[float, str, int]]] = Field(default=None)
    adaboost: Optional[Dict[str, Union[float, str, int]]] = Field(default=None)
    nb: Optional[Dict[str, Union[float, str, int]]] = Field(default=None)
    qda: Optional[Dict[str, Union[float, str, int]]] = Field(default=None)
    xgb: Optional[Dict[str, Union[float, str, int]]] = Field(default=None)


    