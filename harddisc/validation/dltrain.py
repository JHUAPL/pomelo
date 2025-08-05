import os

from distutils.util import strtobool

from pathlib import Path

from typing import Dict, List, Literal, Optional, Tuple, Union
from typing_extensions import Annotated

from pydantic import AfterValidator, BaseModel, conlist, model_validator, Field, ConfigDict

def check_path(value: Path) -> Path:
    if not value.exists():
        raise ValueError(f"{value} topic model output dir does not exist")
    if not value.is_dir():
        raise ValueError(f"{value} topic model output dir does is not a directory")
    if not os.access(str(value), os.R_OK):
        raise ValueError(f"{value} topic model output dir does not have read permissions")
    if not os.access(str(value), os.W_OK):
        raise ValueError(f"{value} topic model output dir does not have write permissions")
    
    return value

def model_change(value: Union[List[str], str]) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if "all" in value:
        return ["bert", "roberta", "xlm", "xlmr", "xlnet"]
    return value

def multimodal_change(value: List[Union[List[str], str]]) -> List[str]:
    for i, item in enumerate(value):
        if isinstance(value, str):
            value[i] = [value[i]]
        if "all" in value[i]:
            value[i] = ["none","concat","mlp_cat","mlp_cat_num","mlp_concat_cat_num","attention","gating","weighted"]
    return value

def ensemble_change(value: List[Union[List[str], str]]) -> List[str]:
    for i, item in enumerate(value):
        if isinstance(value, str):
            value[i] = [value[i]]
        if "all" in value[i]:
            value[i] = ["bagging","fastgeometric","fusion","gradient","snapshot","softgradient","voting","singular"]
    return value

def check_path(value: Path) -> Path:
    if not value.exists():
        raise ValueError(f"{value} DL train output dir does not exist")
    if not value.is_dir():
        raise ValueError(f"{value} DL train output dir does is not a directory")
    if not os.access(str(value), os.R_OK):
        raise ValueError(f"{value} DL train output dir does not have read permissions")
    if not os.access(str(value), os.W_OK):
        raise ValueError(f"{value} DL train output dir does not have write permissions")
    
    return value

class MultimodalModelArguments(BaseModel):
    model_config = ConfigDict(extra='forbid')
    mlp_division: Optional[int] = Field(default=None)
    mlp_dropout: Optional[float] = Field(default=None)
    numerical_bn: Optional[bool] = Field(default=None)
    use_simple_classifier: Optional[bool] = Field(default=None)
    mlp_act: Optional[str] = Field(default=None)
    gating_beta: Optional[float] = Field(default=None)

class ModelArguments(BaseModel):
    model_config = ConfigDict(extra='forbid')
    train_split: Optional[float] = Field(default=None)
    dev_split: Optional[float] = Field(default=None)
    epochs: Optional[int] = Field(default=None)
    lr: Optional[float] = Field(default=None)
    batch_size: Optional[int] = Field(default=None)
    dev_batch_size: Optional[int] = Field(default=None)
    accumulation_steps: Optional[int] = Field(default=None)
    weight_decay: Optional[float] = Field(default=None)
    warmup_steps: Optional[int] = Field(default=None)
    label_smoothing: Optional[float] = Field(default=None)
    cycle: Optional[int] = Field(default=None)
    lr_1: Optional[float] = Field(default=None)
    lr_2: Optional[float] = Field(default=None)
    voting_strategy: Optional[str] = Field(default=None)
    shrinkage_rate: Optional[float] = Field(default=None)
    use_reduction_sum: Optional[bool] = Field(default=None)
    lr_clip: Optional[Union[List[float], Tuple[float]]] = Field(default=None)
    early_stopping_rounds: Optional[int] = Field(default=None)
    n_jobs: Optional[int] = Field(default=None)
    n_estimators: Optional[int] = Field(default=None)
    scheduler: Optional[Dict[str, Union[str, float, int]]] = Field(default=None)
    optimizer: Optional[Dict[str, Union[str, float, int]]] = Field(default=None)
    multimodal: Optional[MultimodalModelArguments] = Field(default=None)

def snapshot_check(value: ModelArguments) -> ModelArguments:
    if value.n_estimators is not None and value.epochs is not None:
        if value.n_estimators % value.epochs != 0:
            raise ValueError(  # noqa: TRY003
                f"Snapshot epochs must be a multiple of n_estimators"
            )
    
    return value

class EnsembleConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    bagging: Optional[ModelArguments] = Field(default=None)
    fastgeometric: Optional[ModelArguments] = Field(default=None)
    fusion: Optional[ModelArguments] = Field(default=None)
    gradient: Optional[ModelArguments] = Field(default=None)
    snapshot: Annotated[Optional[ModelArguments], AfterValidator(snapshot_check)] = Field(default=None)
    softgradient: Optional[ModelArguments] = Field(default=None)
    voting: Optional[ModelArguments] = Field(default=None)
    singular: Optional[ModelArguments] = Field(default=None)


class DLModelConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model_type: Optional[str] = Field(default=None)
    none: Optional[EnsembleConfig] = Field(default=None)
    concat: Optional[EnsembleConfig] = Field(default=None)
    mlp_cat: Optional[EnsembleConfig] = Field(default=None)
    mlp_cat_num: Optional[EnsembleConfig] = Field(default=None)
    mlp_concat_cat_num: Optional[EnsembleConfig] = Field(default=None)
    attention: Optional[EnsembleConfig] = Field(default=None)
    gating: Optional[EnsembleConfig] = Field(default=None)
    weighted: Optional[EnsembleConfig] = Field(default=None)

class DLTrainConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    models: Annotated[Literal["bert", "roberta", "xlm", "xlmr","xlnet", "all"] | conlist(Literal["bert", "roberta", "xlm", "xlmr","xlnet", "all"], min_length=1), AfterValidator(model_change)]
    multimodal: Annotated[Literal["all"] | conlist(Literal["all", "none","concat","mlp_cat","mlp_cat_num","mlp_concat_cat_num","attention","gating","weighted"] | List[Literal["all", "none","concat","mlp_cat","mlp_cat_num","mlp_concat_cat_num","attention","gating","weighted"]], min_length=1), AfterValidator(multimodal_change)]
    ensembles: Annotated[Literal["all"] | conlist(Literal["all", "bagging","fastgeometric","fusion","gradient","snapshot","softgradient","voting","singular"] | List[Literal["all", "bagging","fastgeometric","fusion","gradient","snapshot","softgradient","voting","singular"]], min_length=1), AfterValidator(ensemble_change)]
    offline: Optional[bool] = Field(default=None)
    output_dir: Annotated[Path, AfterValidator(check_path)]
    bert: Optional[DLModelConfig] = Field(default=None)
    roberta: Optional[DLModelConfig] = Field(default=None)
    xlm: Optional[DLModelConfig] = Field(default=None)
    xlmr: Optional[DLModelConfig] = Field(default=None)
    xlnet: Optional[DLModelConfig] = Field(default=None)

    @model_validator(mode='after')
    def check_existence_of_stage(self):
        if len(self.ensembles) != len(self.models):
            raise ValueError(  # noqa: TRY003
                "Must have the same amount of ensemble/none specifications with models"
            )

        if len(self.ensembles) != len(self.multimodal):
            raise ValueError(  # noqa: TRY003
                "Must have the same amount of multimodal specifications with ensembles"
            )

        if len(self.models) != len(self.multimodal):
            raise ValueError(  # noqa: TRY003
                "Must have the same amount of multimodal specifications with models"
            )
        
        return self
    
    @model_validator(mode='after')
    def offline_check(self):
        if self.offline is None:
            try:
                self.offline = bool(strtobool(os.environ["TRANSFORMERS_OFFLINE"]))
            except KeyError:
                self.offline = False

        for model in self.models:
            if self.offline and getattr(self, model) is None:
                raise ValueError(  # noqa: TRY003
                    f"Must specify path to {model} file if offline"
                )
            if self.offline and getattr(self, model) is not None and getattr(self, model).model_type is None:
                raise ValueError(  # noqa: TRY003
                    f"Must specify path to {model} file if offline"
                )

        return self