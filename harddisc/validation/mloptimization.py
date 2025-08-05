import os

from pathlib import Path

from typing import List, Literal, Optional, Tuple, Union
from typing_extensions import Annotated

from pydantic import AfterValidator, BaseModel, conlist, Field, ConfigDict

def ml_model_change(value: Union[List[str], str]) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if "all" in value:
        return ["xgb", "logreg", "svm", "gaussian"]
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
            raise ValueError(f"{path} machine learning optimization dataset does not exist")
        if not path.is_file():
            raise ValueError(f"{path} machine learning optimization dataset is not a file")
        if not os.access(str(path), os.R_OK):
            raise ValueError(f"{path} machine learning optimization dataset does not have read permissions")
        
    return value 

class XGBoostParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    booster: Optional[List[str]] = Field(default=None)
    eta_min: Optional[float] = Field(default=None)
    eta_max: Optional[float] = Field(default=None)
    grow_policy: Optional[List[str]] = Field(default=None)
    gamma_min: Optional[float] = Field(default=None)
    gamma_max: Optional[float] = Field(default=None)
    max_depth_min: Optional[int] = Field(default=None)
    max_depth_max: Optional[int] = Field(default=None)
    min_child_weight_min: Optional[int] = Field(default=None)
    min_child_weight_max: Optional[int] = Field(default=None)
    max_delta_step: Optional[int] = Field(default=None)
    max_delta_step_min: Optional[int] = Field(default=None)
    max_delta_step_max: Optional[int] = Field(default=None)
    subsample_min: Optional[float] = Field(default=None)
    subsample_max: Optional[float] = Field(default=None)
    colsample_bytree_min: Optional[float] = Field(default=None)
    colsample_bytree_max: Optional[float] = Field(default=None)
    colsample_bylevel_min: Optional[float] = Field(default=None)
    colsample_bylevel_max: Optional[float] = Field(default=None)
    colsample_bynode_min: Optional[float] = Field(default=None)
    colsample_bynode_max: Optional[float] = Field(default=None)
    reg_alpha_min: Optional[int] = Field(default=None)
    reg_alpha_max: Optional[int] = Field(default=None)
    reg_lambda_min: Optional[int] = Field(default=None)
    reg_lambda_max: Optional[int] = Field(default=None)
    num_leaves_min: Optional[int] = Field(default=None)
    num_leaves_max: Optional[int] = Field(default=None)
    n_estimators_min: Optional[int] = Field(default=None)
    n_estimators_max: Optional[int] = Field(default=None)
    sample_type: Optional[List[str]] = Field(default=None)
    normalize_type: Optional[List[str]] = Field(default=None)
    rate_drop_min: Optional[float] = Field(default=None)
    rate_drop_max: Optional[float] = Field(default=None)
    skip_drop_min: Optional[float] = Field(default=None)
    skip_drop_max: Optional[float] = Field(default=None)

class SVCParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    C_min: Optional[float] = Field(default=None)
    C_max: Optional[float] = Field(default=None)
    kernel: Optional[List[str]] = Field(default=None)
    degree: Optional[List[int]] = Field(default=None)
    gamma_min: Optional[float] = Field(default=None)
    gamma_max: Optional[float] = Field(default=None)
    coef0_min: Optional[float] = Field(default=None)
    coef0_max: Optional[float] = Field(default=None)

class LogRegParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    penalty: Optional[List[str]] = Field(default=None)
    C_min: Optional[float] = Field(default=None)
    C_max: Optional[float] = Field(default=None)
    max_iter: Optional[List[int]] = Field(default=None)
    l1_ratio_min: Optional[float] = Field(default=None)
    l1_ratio_max: Optional[float] = Field(default=None)

class GPParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kernel: Optional[List[str]] = Field(default=None)

class MachineLearningOptimizationParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    xgb:  Optional[XGBoostParams] = Field(default=None)
    svc:  Optional[SVCParams] = Field(default=None)
    logreg:  Optional[LogRegParams] = Field(default=None)
    gp:  Optional[GPParams] = Field(default=None)


class MLOptimizationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    models: Annotated[Union[
    conlist(Literal["xgb", "logreg", "svm", "gaussian", "all"], min_length=1),
    Literal["xgb", "logreg", "svm", "gaussian", "all"]
], AfterValidator(ml_model_change)]
    datasets: Annotated[Union[List[Path], Path], AfterValidator(ml_file_change)]
    trials: int
    params: Optional[MachineLearningOptimizationParams] = Field(default=None)