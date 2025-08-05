import os

from pathlib import Path

from typing import List, Literal, Optional, Tuple, Union
from typing_extensions import Annotated

from pydantic import AfterValidator, BaseModel, conlist, Field, model_validator, ConfigDict

def check_path(value: Path) -> Path:
    if not value.exists():
        raise ValueError(f"{value} encoding output dir does not exist")
    if not value.is_dir():
        raise ValueError(f"{value} encoding output dir does is not a directory")
    if not os.access(str(value), os.R_OK):
        raise ValueError(f"{value} encoding output dir does not have read permissions")
    if not os.access(str(value), os.W_OK):
        raise ValueError(f"{value} encoding output dir does not have write permissions")

    return value

def embedding_change(value: Union[List[str], str]) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if "all" in value:
        return ["bert", "sent", "tfidf"]
    return value

def reduction_change(value: Union[List[str], str]) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if "all" in value:
        return ["pca", "ae", "vae", "umap"]
    return value

class UMAPConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    components: int
    n_neighbors: Optional[int] = Field(default=None)
    metric: Optional[str] = Field(default=None)
    min_dist: Optional[float] = Field(default=None)

class PCAConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    components: int
    minmax: Optional[bool] = Field(default=None)

class AutoEncoderConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    encoding_layers: Optional[List[int]] = Field(default=None)
    decoding_layers: Optional[List[int]] = Field(default=None)
    train_batch_size: Optional[int] = Field(default=None)
    dev_batch_size: Optional[int] = Field(default=None)
    epochs: Optional[int] = Field(default=None)
    l1: Optional[float] = Field(default=None)
    l2: Optional[float] = Field(default=None)
    lr: Optional[float] = Field(default=None)
    noise: Optional[bool] = Field(default=None)

class BERTConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model_type: str

class SentenceTransformerConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model_type: str

class EncodingConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    embedding_types: Annotated[conlist(Literal["bert", "sent", "tfidf", "all"], min_length=1) | Literal["bert", "sent", "tfidf", "all"], AfterValidator(embedding_change)]
    dimension_reductions: Optional[Annotated[conlist(Literal["pca", "ae", "vae", "umap", "all"], min_length=1) | Literal["pca", "ae", "vae", "umap", "all"], AfterValidator(reduction_change)]] = Field(default=None)
    output_dir: Annotated[Path, AfterValidator(check_path)]
    pca: Optional[PCAConfig] = Field(default=None)
    ae: Optional[AutoEncoderConfig] = Field(default=None)
    vae: Optional[AutoEncoderConfig] = Field(default=None)
    umap: Optional[UMAPConfig] = Field(default=None)
    bert: Optional[BERTConfig] = Field(default=None)
    sent: Optional[SentenceTransformerConfig] = Field(default=None)

    @model_validator(mode='after')
    def existence_check(self):
        if self.dimension_reductions is not None:
            for model in self.dimension_reductions:
                if model == "pca" and getattr(self, "pca") is None:
                    raise ValueError("Must specify pca components")
                if model == "umap" and getattr(self, "umap") is None:
                    raise ValueError("Must specify umap components")
        
        return self



    
