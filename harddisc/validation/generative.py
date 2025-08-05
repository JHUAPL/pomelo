import os

from pathlib import Path

from typing import Any, List, Literal, Optional, Tuple, Union, Dict
from typing_extensions import Annotated

from pydantic import AfterValidator, BaseModel, conlist, Field, model_validator, ConfigDict

def model_change(value: Union[List[str], str]) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if "all" in value:
        return ["openaichat", "openai", "hfoffline", "hfonline"]
    return value

def model_change_inner(value: Union[List[str], str]) -> List[str]:
    if isinstance(value, str):
        value = [value]
        
    if len(value) == 0:
        raise ValueError(f"Not enough models")
    
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

class OpenAIChatMessagesConfig(BaseModel):
    role: str
    name: Optional[str] = Field(default=None)

class OpenAIChatConfig(BaseModel):
    model_config = ConfigDict(extra='allow')
    model_names: Annotated[Union[List[str], str], AfterValidator(model_change_inner)]
    
class OpenAIConfig(BaseModel):
    model_config = ConfigDict(extra='allow')
    model_names: Annotated[Union[List[str], str], AfterValidator(model_change_inner)]
    
class HFOfflineConfig(BaseModel):
    model_config = ConfigDict(extra='allow')
    model_names: Annotated[Union[List[str], str], AfterValidator(model_change_inner)]
    

class GenerativeConfig(BaseModel):
    models: Annotated[Union[
    conlist(Literal["openaichat", "openai", "hfoffline", "hfonline", "all"], min_length=1),
    Literal["all", "openaichat", "openai", "hfoffline", "hfonline"]
], AfterValidator(model_change)]
    prompt: str
    label_set: Dict[str, Any]
    output_dir: Annotated[Path, AfterValidator(check_path)]
    openaichat: Optional[OpenAIChatConfig] = Field(default=None)
    openai: Optional[OpenAIConfig] = Field(default=None)
    hfoffline: Optional[HFOfflineConfig] = Field(default=None)

    @model_validator(mode='after')
    def existence_check(self):
        if "openai" in self.models and getattr(self, "openai") is not None:
            self.openai = self.openai_model_names_change(getattr(self, "openai"))
        elif "openai" in self.models and getattr(self, "openai") is None:
            raise ValueError("Need to have openai model specified to work")
        
        if "openaichat" in self.models and getattr(self, "openaichat") is not None:
            self.openaichat = self.openai_chat_model_names_change(getattr(self, "openaichat"))
        elif "openaichat" in self.models and getattr(self, "openaichat") is None:
            raise ValueError("Need to have openai chat model specified to work")
        
        if "hfoffline" in self.models and getattr(self, "hfoffline") is not None:
            self.hfoffline = self.hfoffline_model_names_change(getattr(self, "hfoffline"))
        elif "hfoffline" in self.models and getattr(self, "hfoffline") is None:
            raise ValueError("Need to have hfoffline model specified to work")
        
        return self

    def openai_model_names_change(self, model_args: OpenAIConfig) -> List[str]:
        for model in model_args.model_names:
            if getattr(model_args, model, None) is None:
                raise ValueError(f"Must specify {model} for openai")
            elif getattr(model_args, model, None) is not None and getattr(model_args, model, None).get("batch_size", None) is None:
                raise ValueError(f"Must specify batch size {model} for openai")

        return model_args

    def openai_chat_model_names_change(self, model_args: OpenAIChatConfig) -> List[str]:
        for model in model_args.model_names:
            if getattr(model_args, model, None) is not None:
                if getattr(model_args, model, None).get("messages", None) is not None:
                    if getattr(model_args, model, None).get("messages", None).get("role", None) is None:
                        raise ValueError(f"Must specify messages role {model} for openai chat")
                else:
                    raise ValueError(f"Must specify messages {model} for openai chat")
            else:
                raise ValueError(f"Must specify {model} for openai chat")
                    
        return model_args

    def hfoffline_model_names_change(self, model_args: HFOfflineConfig) -> List[str]:
        for model in model_args.model_names:
            if getattr(model_args, model, None) is None:
                raise ValueError(f"Must specify {model} for hfoffline")
            elif getattr(model_args, model, None) is not None and getattr(model_args, model, None).get("batch_size", None) is None:
                raise ValueError(f"Must specify batch size {model} for hfoffline")

        return model_args