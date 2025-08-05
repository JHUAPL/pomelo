from typing import List, Literal, Optional, Tuple, Union
from typing_extensions import Annotated

from pydantic import AfterValidator, BaseModel, conlist, model_validator

from harddisc.validation.dataset import DatasetConfig
from harddisc.validation.dloptimization import DLOptimizationConfig
from harddisc.validation.dltrain import DLTrainConfig
from harddisc.validation.encoding import EncodingConfig
from harddisc.validation.generative import GenerativeConfig
from harddisc.validation.mloptimization import MLOptimizationConfig
from harddisc.validation.mltrain import MLTrainConfig
from harddisc.validation.topicmodel import TopicModelConfig

def stages_validator(value: Union[List[str], str]) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if value == "all":
        return [
                "encoding",
                "mltrain",
                "dltrain",
                "topicmodel",
                "dloptimization",
                "mloptimization",
                "generative",
            ]
    return value

class POMELOConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    dataset: DatasetConfig
    stages: Annotated[Union[
    conlist(List[Literal["encoding","mltrain","dltrain","topicmodel","dloptimization","mloptimization","generative","all"]], min_length=1),
    Literal["encoding","mltrain","dltrain","topicmodel","dloptimization","mloptimization","generative","all"]
], AfterValidator(stages_validator)]
    random_seed: int
    encoding: Optional[EncodingConfig]
    mltrain: Optional[MLTrainConfig]
    dltrain: Optional[DLTrainConfig]
    topicmodel: Optional[TopicModelConfig]
    mloptimization: Optional[MLOptimizationConfig]
    dloptimization: Optional[DLOptimizationConfig]
    generative: Optional[GenerativeConfig]

    @model_validator(mode='after')
    def check_existence_of_stage(self):
        for config_stage in self.stages:
            if getattr(self, config_stage) is None:
                raise ValueError(  # noqa: TRY003
                        f"{config_stage} is not in current set of stages ({self.stages})"
                    )
        return self
        