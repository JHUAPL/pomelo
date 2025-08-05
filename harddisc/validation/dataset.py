import os

from typing import List, Literal, Optional, Tuple, Union, Set
from typing_extensions import Annotated

from pathlib import Path

import pandas as pd

from pydantic import AfterValidator, BaseModel, conlist, model_validator, Field, ConfigDict


class JargonConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    path: Path
    jargon_column: str
    expanded_column: str

    @model_validator(mode='after')
    def csv_checking(self):
        if not self.path.exists():
            raise ValueError(f"{self.path} jargon dataset does not exist")
        if not self.path.is_file():
            raise ValueError(f"{self.path} overall dataset is not a file")
        if not os.access(str(self.path), os.R_OK):
            raise ValueError(f"{self.path} jargon dataset does not have read permissions")
        
        try: 
            df = pd.read_csv(self.path, nrows=1)
        except (FileNotFoundError, pd.EmptyDataError, pd.errors.ParserError) as e:
            raise ValueError(f"Obtained error {e} while opening {self.path} using pandas")
        
        dataset_columns = set(df.columns)

        if self.expanded_column not in dataset_columns:
            raise ValueError(  # noqa: TRY003
                f"Column for expanded jargon: {self.expanded_column} does not exist in {self.path}"
            )
        if self.jargon_column not in dataset_columns:
            raise ValueError(  # noqa: TRY003
                f"Column for expanded jargon: {self.jargon_column} does not exist in {self.path}"
            )
        
        return self


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    dataset_path: Path
    free_text_column: str
    prediction_column: str
    categorical_columns: List[str]
    date_columns: List[str]
    numerical_columns: List[str]
    jargon: Optional[JargonConfig] = Field(default=None)

    @model_validator(mode='after')
    def csv_checking(self):
        if not self.dataset_path.exists():
            raise ValueError(f"{self.dataset_path} overall dataset does not exist")
        if not self.dataset_path.is_file():
            raise ValueError(f"{self.dataset_path} overall dataset is not a file")
        if not os.access(str(self.dataset_path), os.R_OK):
            raise ValueError(f"{self.dataset_path} overall dataset does not have read permissions")
        
        try: 
            df = pd.read_csv(self.dataset_path, nrows=1)
        except (FileNotFoundError, pd.EmptyDataError, pd.errors.ParserError) as e:
            raise ValueError(f"Obtained error {e} while opening {self.dataset_path} using pandas")
        
        dataset_columns = set(df.columns)

        if self.free_text_column not in dataset_columns:
            raise ValueError(  # noqa: TRY003
                f"Free Text Column: {self.free_text_column} does not exist in {self.dataset_path}"
            )

        if self.prediction_column not in dataset_columns:
            raise ValueError(  # noqa: TRY003
                f"Prediction Column: {self.prediction_column} does not exist in {self.dataset_path}"
            )
        
        self.check_columns(self.categorical_columns, dataset_columns)
        self.check_columns(self.numerical_columns, dataset_columns)
        self.check_columns(self.date_columns, dataset_columns)

        return self
    
    def check_columns(self, columns: List[str], dataset_columns: Set[str]):
        """Helper function to check if a list of columns are in the dataset provided

        Parameters
        ----------
        column_type : str
            category of column that needs to be checked (used for error purposes)
        dataset_columns : Set[str]
            columns that fit that category

        Raises
        ------
        ValueError
            At least one column provided is not in the dataset
        """
        if len(columns) != 0 and not set(
            columns
        ).issubset(dataset_columns):
            missing_cols = set(columns).difference(dataset_columns)
            raise ValueError(  # noqa: TRY003
                f"{missing_cols} do not exist in {self.dataset_path}"
            )