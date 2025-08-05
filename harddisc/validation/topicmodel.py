import os

from pathlib import Path

from typing import List, Literal, Optional, Tuple, Union
from typing_extensions import Annotated

from pydantic import AfterValidator, BaseModel, conlist, Field, ConfigDict

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



def topic_model_change(value: Union[List[str], str]) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if "all" in value:
        return ["bertopic", "lda"]
    return value

class BERTopicConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    top_n_words: Optional[int] = Field(default=None)
    nr_topics: Optional[int] = Field(default=None)
    min_topic_size: Optional[int] = Field(default=None)
    n_gram_range: Optional[Tuple[int, int]] = Field(default=None)
    model: Optional[str] = Field(default=None)


class TopicModelConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    models: Annotated[Union[
    conlist(Literal["bertopic", "lda", "all"], min_length=1),
    Literal["all", "bertopic", "lda"]
], AfterValidator(topic_model_change)]
    output_dir: Annotated[Path, AfterValidator(check_path)]
    bertopic: Optional[BERTopicConfig] = Field(default=None)
   




