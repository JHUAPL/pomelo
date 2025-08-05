import os
from distutils.util import strtobool

from typing import List, Literal, Optional, Tuple, Union
from typing_extensions import Annotated

from pathlib import Path

from pydantic import AfterValidator, BaseModel, conlist, model_validator, Field, ConfigDict

def dloptimization_model_change(value: Union[List[str], str]) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if "all" in value:
        return ["bert", "roberta", "xlm", "xlmr","xlnet"]
    return value

class DeepLearningOptimizationParams(BaseModel):
    epoch_min: Optional[int] = Field(default=None)
    epoch_max: Optional[int] = Field(default=None)
    lr_min: Optional[float] = Field(default=None)
    lr_max: Optional[float] = Field(default=None)
    weight_decay_min: Optional[float] = Field(default=None)
    weight_decay_max: Optional[float] = Field(default=None)
    accumulation_steps: Optional[List[int]] = Field(default=None)
    scheduler: Optional[List[str]] = Field(default=None)
    warmup_steps_min: Optional[int] = Field(default=None)
    warmup_steps_max: Optional[int] = Field(default=None)
    batch_size: Optional[List[int]] = Field(default=None)
    label_smoothing_min: Optional[float] = Field(default=None)
    label_smoothing_max: Optional[float] = Field(default=None)
    mlp_division_min: Optional[int] = Field(default=None)
    mlp_division_max: Optional[int] = Field(default=None)
    mlp_dropout_min: Optional[float] = Field(default=None)
    mlp_dropout_max: Optional[float] = Field(default=None)
    combine_feat_method: Optional[List[str]] = Field(default=None)
    gating_beta_min: Optional[float] = Field(default=None)
    gating_beta_max: Optional[float] = Field(default=None)
    ensemble_method: Optional[List[str]] = Field(default=None)
    n_estimators_min: Optional[int] = Field(default=None)
    n_estimators_max: Optional[int] = Field(default=None)
    epochs_snapshot_min: Optional[int] = Field(default=None)
    epochs_snapshot_max: Optional[int] = Field(default=None)

class ModelArgsConfig(BaseModel):
    model_type: Optional[str] = Field(default=None)
    train_split: Optional[float] = Field(default=None)
    dev_split: Optional[float] = Field(default=None)

class DLOptimizationConfig(BaseModel):
    models: Annotated[Literal["bert", "roberta", "xlm", "xlmr","xlnet", "all"] | conlist(Literal["bert", "roberta", "xlm", "xlmr","xlnet", "all"], min_length=1), AfterValidator(dloptimization_model_change)]
    trials: int
    offline: Optional[bool] = Field(default=None)
    bert: Optional[ModelArgsConfig] = Field(default=None)
    roberta: Optional[ModelArgsConfig] = Field(default=None)
    xlm: Optional[ModelArgsConfig] = Field(default=None)
    xlmr: Optional[ModelArgsConfig] = Field(default=None)
    xlnet: Optional[ModelArgsConfig] = Field(default=None)
    params: Optional[DeepLearningOptimizationParams] = Field(default=None)

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