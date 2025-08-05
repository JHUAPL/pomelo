# HARDDISC-POMELO: A Multimodal AutoML/DL Suite for Classification

# What does it do
Multimodal machine and deep learning classification is hard. Putting together text, numbers, and categories? How on earth do you do that?\
With POMELO!

4 Different ways of dealing with classification tasks

1. Unsupervised methods
    - BERTopic
    - LDA
2. ML Supervised methods
    - sklearn classifiers 
    - xgboost
3. DL Supervised methods
    - BERT, RoBERTa, XLM, XLMR, XLNET
4. DL Multimodal methods
    - Combine categorical and numerical data with your text!
5. DL Ensemble methods
    - Add the power of more models!
6. Generative models
    - ask them to summarize
    - ask them why something happened
    - ask them what happened

Different ways of encoding text and dimension reduction!
1. Encoding
    - BERT
    - Sentence Transformer
    - TFIDF
    - Bag of Words
2. Dimension Reduction
    - Autoencoder
    - Variational Autoencoder
    - PCA
    - UMAP

Descriptive plots for your visualization
1. TSNE
    - see how your data was encoded
2. ROC Curve
    - see the performance of your model
3. Metric Plots
    - compare performances across datasets
4. Topics
    - see whats important to your text

# Setup

## Quick Setup
Installing is really simple!\
Tested on Python 3.10.16 and Ubuntu 22.04\

Set up venv
```
python3.10 -m venv env
source env/bin/activate
```

Clone repo and navigate to it
```
git clone https://github.com/JHUAPL/pomelo.git
cd pomelo
```
Download the zip of the Multimodal-Toolkit found here: https://github.com/georgian-io/Multimodal-Toolkit _(we do not pip install this package from pypi because the versions there do not have certain classes inside model.tabular_transformers.py, such as XLMWithTabular)_.

Drag the folder within that zip into your pomelo repo.

Navigate to the file:

```
Multimodal-Toolkit/multimodal_transformers/model/__init__.py

```

To the following section:
```
from .tabular_transformers import (
    BertWithTabular,
    RobertaWithTabular,
    DistilBertWithTabular,
    LongformerWithTabular
)


__all__ = [
    "TabularFeatCombiner",
    "TabularConfig",
    "AutoModelWithTabular",
    "BertWithTabular",
    "RobertaWithTabular",
    "DistilBertWithTabular",
    "LongformerWithTabular"
]
```
Change it to:

```
from .tabular_transformers import (
    BertWithTabular,
    RobertaWithTabular,
    DistilBertWithTabular,
    XLMWithTabular,
    XLMRobertaWithTabular,
    XLNetWithTabular
)


__all__ = [
    'TabularFeatCombiner',
    'TabularConfig',
    'AutoModelWithTabular',
    'BertWithTabular',
    'RobertaWithTabular',
    'DistilBertWithTabular',
    'XLMWithTabular',
    'XLMRobertaWithTabular',
    'XLNetWithTabular'
]
```
navigate back to the repo and install Packages via:

```
pip install .
```

## Offline Setup
If you are working in an offline environment, these are the steps you must follow
1. Install pre-requisites if they are not already in the environment
2. Download the wheelhouse zip from the releases
3. Put wheelhouse on flash drive/CD and upload to the environment
4. ```unzip [pomelo-tar|pomelo-whl].zip```
5. ```python3.9 -m venv env```
6. ```source env/bin/activate```
7. ```pip install [pomelo-tar|pomelo-whl]/harddisc-x.x.x.[tar.gz|-py3-none-any.whl] --no-index --find-links [pomelo-whl|pomelo-tar]/```
8. To run DVC without git (you downloaded the repo straight up), run ```dvc config core.no_scm true``` to prevent the error


## Huggingface Offline Setup

1. To work with models offline, please set ```TRANSFORMERS_OFFLINE=1``` in `.env`
2. Run ```source .env``` to update env variables
3. To get models go to the [Model Hub](https://huggingface.co/models)
4. Find your favorite model, and go to the files
  - For example, to run the entire pipeline E2E we use
    - BERT: bert-base-uncased
    - ROBERTA: roberta-base
    - XLNET: xlnet-base-cased
    - XLMR: xlm-roberta-base
    - XLM: xlm-mlm-en-2048
5. Click the downarrow on 
    - config.json
    - pytorch_model.bin
    - tokenizer.json
    - tokenizer_config.json
    - vocab.txt 
6. Put them all into the folder
  - We recommend putting all your models in folder named `models` at the repo level
7. Alternative way to download model and tokenizer (probably better)
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")

tokenizer.save_pretrained("./your/path/bigscience_t0")
model.save_pretrained("./your/path/bigscience_t0")
```

## Dev Set up
1. Do all the steps above for quick/offline setup
2. Run ```pip install -r dev-requirements.txt``` to install dev reqs
3. Run ```python -m pytest test/``` to check if your system installed everything correctly
4. Run ```python -m pytest test_gpu/``` to check if the LLM part is installed and can run correctly 


## Running harddisc
This is the command to run the suite
```python -m harddisc -c /path/to/params.yaml```

The bare minimum to `params.yaml` file to run the entire pomelo suite using default settings is 
```yaml
stages: "all"
random_seed: 666
dataset:
  dataset_path: "path/to/csv.csv"
  free_text_column: "column_that_has_text"
  prediction_column: "column_that_has_class"
  categorical_columns: ["column_that_is_categorical_1", "column_that_is_categorical_2", "column_that_is_categorical_3"]
  date_columns: ["column_that_has_a_date"]
  numerical_columns: ["column_that_is_numbers1", "column_that_is_numbers2"]
encoding:
  embedding_types: "all"
  dimension_reductions: "all"
  output_dir: "test"
  pca: 
    components: 70
  umap: 
    components: 70
```
After encoding then you can run
```yaml
stages: "all"
random_seed: 666
dataset:
  dataset_path: "path/to/csv.csv"
  free_text_column: "column_that_has_text"
  prediction_column: "column_that_has_class"
  categorical_columns: ["column_that_is_categorical_1", "column_that_is_categorical_2", "column_that_is_categorical_3"]
  date_columns: ["column_that_has_a_date"]
  numerical_columns: ["column_that_is_numbers1", "column_that_is_numbers2"]
mltrain:
  models: "all"
  datasets: "./data/processed/"
  output_dir: "test"
dltrain:
  models: "all"
  multimodal: ["all", "all", "all", "all", "all"]
  ensembles: ["all", "all", "all", "all", "all"]
  output_dir: "test"
topicmodel:
  models: "all"
  output_dir: "test"
mloptimization:
  models: "all"
  datasets: "./data/processed/"
  trials: 30
dloptimization:
  models: "all"
  trials: 30
generative:
  prompt: "{column_that_is_categorical_1} {column_that_has_text} Question to give model"
  models: "all"
  output_dir: "test"
  label_set:
    "yes" : true
    "no" : false
  hfoffline: 
    model_names: ["model1", "model2"]
    model1:
      batch_size: 1
    model2:
      batch_size: 1
  openaichat:
    model_names : ["model1", "model2"]
      model1:
        messages: 
          role: "role1"
      model2:
        messages: 
          role: "role2"
  openai:
    model_names : ["model1", "model2"]
    model1:
      batch_size: 1
    model2:
      batch_size: 1
  hfonline:
    model_names : ["model1", "model2"]
```

# Examples

More extensive examples can be seen in the ```examples/``` folder. There are 7 examples provided. 

- `examples/example_dloptimization.yaml`
  - example file to optimizes the hyperparameters of deep learning models like BERT and RoBERTa on the example dataset
- `examples/example_dltrain.yaml`
  - example file to train classical machine learning models like BERT and RoBERTa on the example dataset
- `examples/example_encoding.yaml`
  - example file to encode text using BERT and TF-IDF
- `examples/example_generative.yaml`
  - example file to do zeroshot classification with LLMs like distilgpt2 and gpt-neo-125m
- `examples/example_mloptimization.yaml`
  - example file to optimizes the hyperparameters of classical machine learning models like XGBoost and Logistic Regression on the example dataset
- `examples/example_mltrain.yaml`
  - example file to train classical machine learning models like XGBoost and Naive Bayes on the example dataset
- `examples/example_topicmodelling.yaml`
  - example file to run bertopic and lda on the exampledataset

## Running Examples
To run the examples
```python -m harddisc -c examples/example_<PIPELINE_STAGE>.yaml```


# Parameters

Config divided into 8 sections


YAML will look like this
```yaml
dataset:
  ...
encoding:
  ...
mltrain:
  ...
mloptimization:
  ...
dltrain:
  ...
dloptimization:
  ...
topicmodel:
  ...
generative:
  ...
```
## NB
If you are using a float, you can either do 0.001 or 1.0E-3 (exactly)

## Overall Required Params
- Required Params:
- `stages`: List[str]/str
  - which stages do you want to run
  - Supported stages (must be in list): "encoding", "mltrain", "dltrain", "topicmodel", "optimization"
  - Supported keywords (must be string): "all"
- `random_seed`: int
  - random seed
## Dataset
- Must include this in every params
- Required Params: 
  - `dataset_path`: str
    - path to csv dataset
  - `free_text_column`: str
    - free text column name in dataeset
  - `prediction_column`: str
    -  class column name in dataset
  - `categorical_columns`: List[str]
    - columns that are categorical data
  - `date_columns`: List[str]
    - columns that are date data
  - `numerical_columns`: List[str]
    - columns that are pure number columns
- Optional 
  - `jargon`
    - Required params:
      - `path`: str
         - valid path to valid csv with jargon expansion pairs
      - `jargon_column`: str
        - valid column of csv that contains the jargon to replace 
      - `expanded_column`: str
        - valid column of csv that contains the words to replace the jargon
## Encoding
  - Required if stage is provided in pipeline setup
  - Required Params
    - `embedding_types`: List[str]/str
      - Supported (must be in list): "bert", "tfidf", "sent"
      - Supported keywords (must be string): "all"
      - Each embedding type can have further options
    - `output_dir`: str
      - Output for encoding
  - Optional Params
    - `bert`
      - Required Params: None
      - Optional Params
        - `model_type`: str
          - model type from huggingface hub or path to folder containing tokenizer and model
          - default: "bert-base-cased"
    - `tfidf`
      - No additional params at this time
    - `sent`
      - Required Params: None
      - Optional Params
        - `model_type`: str
          - model type from ["Small", "Medium", "Large", "XLarge"] hub or path to folder containing tokenizer and model
          - default: "Small"
    - `dimension_reductions` : List[str]/str
      - Supported reductions (must be in list): "pca", "ae", "vae", "umap"
      - Supported keywords (must be string): "all"
    - `pca`
      - Must be included if using PCA 
      - Required Params
        - `components`: int 
        - how many principal components 
      - Optional Params
        - `minmax`: bool
        - Whether to use minmax or standard scaler
        - default = False
    - `ae`/`vae`
      - Required Params: None
      - Optional Params:
        - `encoding_layers`: List[int]
          - Must start with 0
          - Default: [0, 300, 200]
        - `decoding_layers`: List[int]
          - Must end with 0 
          - Default: [200, 300, 0]
        - `train_batch_size`: int
          - Default: 10
        - `dev_batch_size`: int
          - Default: 10
        - `epochs`: int
          - Default 100
        - `l1`: float
          - L1 Penalty
          - Default: 1
        - `l2`: float
          - L2 Penalty (weight decay)
          - Default: 0
        - `lr`: float 
          - learning rate
          - Default: 1
        - `noise`: bool
          - Whether to noise the input with standard normal 
          - Default: False
    - `umap`
      - Must be included if using UMAP
      - Required Params
        - `components` : int
          - how many dimensions after reduction
      - Optional Params
        - `n_neighbors` : int
        - `metric`: str
        - `min_dist`: float
        - For explanations and supported metrics refer to UMAP Python Package
## MLTrain
  - Required if stage is provided in pipeline setup
  - Required Params
    - `models`: List[str]/str
      - Supported (must be in list): "logreg","knn","svm","gaussian","tree","rf","nn","adaboost","nb","qda","xgb"
      - Supported keywords (must be string): "all"
      - Each model type can have further options
    - `datasets` : List[str]/str
      - Can either be 
        - A single file as a string 
          - "/data/processed/tfidf.joblib"
        - A list of files as a list of strings
          - ["/data/processed/tfidf.joblib", "/data/processed/bert.joblib"]
        - A folder of files
          - "/data/processed/"
    - `output_dir`: str
      - Output for training metrics, plots, and files
  - Optional Params
    - `train_split`: float
      - Percentage of dataset dedicated to training
      - Default 0.8
    - Hyperparams
      - To add hyperparams add a line with the abbreviation for the model
      - indented below it put the hyperparams
      - Additional hyperparameters for models can be found in 
## DLTrain 
  - Required if stage is provided in pipeline setup
  - Required Params
    - `models`: List[str]/str
      - Supported (must be in list): "bert", "roberta", "xlm", "xlmr", "xlnet"
      - Supported keywords (must be string): "all" 
    - `multimodal`: List[List[str]]/List[str]
      - Supported (must be in list): "none", "concat", "mlp_cat", "mlp_cat_num", "mlp_concat_cat_num", "attention", "gating", "weighted"
      - Supported keywords (must be string): "all" 
    - `ensembles`: List[List[str]]/List[str]
      - Supported (must be in list): "bagging","fastgeometric","fusion","gradient","snapshot","softgradient","voting","singular"
      - Supported keywords (must be string): "all" 
    - `output_dir`: str
      - Output for training metrics, plots, and files
    
  - Optional Params
    - `bert`: 
      - default `model_type`: bert-base-uncased
    - `roberta`:
      - default `model_type`: robert-base
    - `xlm`:
      - default `model_type`: xlx-mlm-en-2048
    - `xlmr`:
      - default `model_type`: xlm-roberta-base
    - `xlnet`:
      - default `model_type`: xlnet-base-cased
  - Training/Model Args:
    - If you need to add extra parameters, you must specify which model, which multimodal, and which ensemble you want to change the parameters for. For example
    ```yaml
    bert:
      model_type:
      gating:
        bagging:
          <PLACE ARGS FOR GATING BERT WITH BAGGING ENSEMBLE>
    ``` 
  - Singular model args
      - `scheduler`: str 
        - lr scheduler type
        - more info here
        - https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.SchedulerType
        - default: "linear"
      - `train_split`: float
        - Percentage of dataset dedicated to training 
        - Default: 0.8
      - `dev_split`: float
        - Percentage of dataset dedicated to training
        - Default: 0.1
      - `epochs`: int 
        - default: 3
      - `batch_size`: int 
        - train batch size
        - default: 16
      - `dev_batch_size`: int 
        - dev and test batch size
        - default: 32
      - `accumulation_steps`: int 
        - how many batchs  of a dataset to wait before running backprop (effectively multiples batch size),
        - default: 1
      - `lr`: float 
        - learning rate
        - default: 2e-5
      - `weight_decay`: float 
        - weight decay/l2 loss
        - default: 0.0001
      - `warmup_steps`: int 
        - lr scheduler warm up steps
        - default: 100
      - `multimodal` : Dict[str, Any]
        - args for multimodal model
        - explained below
  - Ensemble model args
    - `train_split`: float
      - Percentage of dataset dedicated to training 
      - Default: 0.8
    - `dev_split`: float
      - Percentage of dataset dedicated to training
      - Default: 0.1
    - `epochs`
      - default: 3
    - `batch_size`
      - train batch size
      - default: 16
    - `dev_batch_size`
      - dev and test batch size
      - default: 32
    - `n_estimators`: int
      - Number of estimators to ensemble
      - default : 3
    - `n_jobs`: int
      - Number of parallel processes
      - default: 1
    - `scheduler`: Dict[str, Any]
      - Learning rate scheduler
      - Required args:
        - name: str
          - Name of scheduler must be from [here](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)
      - Optional args
        - Args for scheduler: Any
          - Any of the args from the schedulers 
          - Cannot do lambdas as it will introduce vulnerabilities
      - default: none
    - `optimizer`
      - Optimizer for ensemble
      - Required args
        - name: str
          - name of optimizer must be from [here](https://pytorch.org/docs/stable/optim.html#algorithms)
      - Optional args
        - Args for optimizer: Any
          - Any of the args from the optimizers
      - default: 
          ```yaml
          name: Adam 
          lr : 2e-5 
          weight_decay: 0.001
          ```
    - `multimodal` : Dict[str, Any]
      - args for multimodal model
      - explained below
  - Model specific ensemble args
    - `cycle` : int
      - Fast geometric only
      - The number of cycles used to build each base estimator in the ensemble
      - default: 4
    - `lr_1`: float
      - Fast geometric only
      - ``alpha_1`` in original paper used to adjust the learning rate, also
        serves as the initial learning rate of the internal optimizer
      - default: 5e-2
    - `lr_2`
      - Fast geometric only
      - `alpha_2` in original paper used to adjust the learning rate, also
        serves as the smallest learning rate of the internal optimizer
      - default: 1e-4
    - `voting_strategy`: str
      - Voting ensemble and Snapshot ensemble
      - Must be from the list: "soft", "hard"
      - Whether to average the probabilities (soft) or take a majority vote (hard)
      - default: soft
    - `shrinkage_rate`: float
      - Gradient boosting and soft gradient
      - the shrinkage rate used in gradient boosting
      - default: 1.0
    - `use_reduction_sum`: bool
      - Gradient boosting and soft gradient
      - Whether to set ``reduction="sum"`` for the internal mean squared
        error used to fit each base estimator.
      - default: True
    - `lr_clip` : List[float] | Tuple[float]
        - Snapshot ensemble only
        - Specify the accepted range of learning rate. When the learning rate
        determined by the scheduler is out of this range, it will be clipped
        - The first element should be the lower bound of learning rate
        - The second element should be the upper bound of learning rate
    - `early_stopping_rounds`: int
      - Gradient boosting only
      - Specify the number of tolerant rounds for early stopping. When the
        validation performance of the ensemble does not improve after
        adding the base estimator fitted in current iteration, the internal
        counter on early stopping will increase by one. When the value of
        the internal counter reaches `early_stopping_rounds`, the
        training stage will terminate instantly
      - default: 2
  - Multimodal args
    - `mlp_division`: int
      - how much to decrease each MLP dim for each additional layer
      - default: 4
    - `mlp_dropout`: float 
      - dropout ratio used for MLP layers
      - default: 0.1
    - `numerical_bn`: bool 
      - whether to use batchnorm on numerical features
      - Warning you cannot use batch size 1 with batch norm on. We will turn it off and warn you 
      - default: True
    - `use_simple_classifier`: bool
      - whether to use single layer or MLP as final classifier
      - default: True
    - `mlp_act`: str
      - the activation function to use for finetuning layers
      - Must be from the list: "relu", "prelu", "sigmoid", "tanh", "linear"
      - default: 'relu'
    - `gating_beta`: float
      - the beta hyperparameters used for gating tabular data see the paper Integrating Multimodal Information in Large Pretrained Transformers for details
      - default: 0.2

## MLOptimization
  - Required if stage is provided in pipeline setup
  - Required Params
    - `models`: List[str]/str
      - Supported (must be in list): "xgb", "logreg", "svm", "gaussian"
      - Supported keywords (must be string): "all"
    - `datasets` : List[str]/str
      - Can either be 
        - A single file as a string 
          - "/data/processed/tfidf.joblib"
        - A list of files as a list of strings
          - ["/data/processed/tfidf.joblib", "/data/processed/bert.joblib"]
        - A folder of files
          - "/data/processed/"
    - `trials`: int
      - how many optimization steps you want to do 
  - Optional Params
    - `params`
      - `xgb`
        - `booster`: List[str], default=["gbtree", "gblinear", "dart"]
        - `eta_min`: float, default=0.1
        - `eta_max`: float, default=1
        - `grow_policy`: List[str], default=["depthwise", "lossguide"]
        - `gamma_min`: float, default=1
        - `gamma_max`: float, default=9
        - `max_depth_min`: int, default=3
        - `max_depth_max`: int, default=18
        - `min_child_weight_min`: int, default=0
        - `min_child_weight_max`: int, default=10
        - `max_delta_step`: int, default=0,  
        - `max_delta_step_min`: int, default=0
        - `max_delta_step_max`: int, default=10
        - `subsample_min`: float, default=0.1
        - `subsample_max`: float, default=1
        - `colsample_bytree_min`: float, default=0.1
        - `colsample_bytree_max`: float, default=1
        - `colsample_bylevel_min`: float, default=0.1
        - `colsample_bylevel_max`: float, default=1
        - `colsample_bynode_min`: float, default=0.1
        - `colsample_bynode_max`: float, default=1
        - `reg_alpha_min`: int, default=40
        - `reg_alpha_max`: int, default=180
        - `reg_lambda_min`: int, default=0
        - `reg_lambda_max`: int, default=1
        - `num_leaves_min`: int, default=1
        - `num_leaves_max`: int, default=10
        - `n_estimators_min`: int, default=100
        - `n_estimators_max`: int, default=10000
        - `sample_type`: List[str], default=["uniform", "weighted"]
        - `normalize_type`: List[str], default=["tree", "forest"]
        - `rate_drop_min`: float, default=1e-8
        - `rate_drop_max`: float, default=1.0
        - `skip_drop_min`: float, default=1e-8
        - `skip_drop_max`: float, default=1.0
      - `logreg`
        - `penalty`: List[str], default=["none", "l2", "l1", "elasticnet"]
        - `C_min`: float, default=0.1
        - `C_max`: float, default=1000
        - `max_iter`: List[int], default = [100, 150, 200, 250, 300, 500]
        - `l1_ratio_min`: float, default=0
        - `l1_ratio_max`: float, default=1
      - `svc`
        - `C_min`: float, default=0.001
        - `C_max`: float, default=1000.0
        - `kernel`: List[str], default=["linear", "poly", "rbf", "sigmoid"]  
        - `degree`: List[int], default= [3, 4, 5, 6]
        - `gamma_min`: float, default=0.001
        - `gamma_max`: float, default=1000.0
        - `coef0_min`: float, default=0.0
        - `coef0_max`: float, default=10.0
      - `gp`
        - `kernel`: List[str], default = ["matern", "rbf", "rq", "ess", "dp"]

## Topic Model
  - Required if stage is provided in pipeline setup
  - Required Params
    - `models`: List[str]/str
      - Supported (must be in list): "lda","bertopic"
      - Supported keywords (must be string): "all" 
    - `output_dir`: str
      - Output for training metrics, plots, and files
  - Optional Params
    - `bertopic`: 
      - Required Params
        - `model`: str
          - Path to sentence transformer model directory
          - If you timeout this is probably the best way of using bertopic
        - `top_n_words`: int
          - The number of words per topic to extract
          - default: 5
        - `nr_topics`: int | str
          - Specifying the number of topics will reduce the initial number of topics to the value specified. 
          - default: "auto"
        - `min_topic_size`: int
          - The minimum size of the topic.
          - default: 20
        - `n_gram_range`: List[int, int]
          - The n-gram range for the CountVectorizer.
          - Default [1,2]
        - `model`: str
          - Use a custom embedding model. The following backends are currently supported * SentenceTransformers * Flair * Spacy * Gensim * 
          - default: none

## DLOptimization
  - Required if stage is provided in pipeline setup
  - Required Params
    - `models`: List[str]/str
      - Supported (must be in list): "xgb", "logreg", "svm", "gaussian"
      - Supported keywords (must be string): "all"
    - `trials`: int
      - how many optimization steps you want to do 
    - `offline`: bool
      - Whether training models from online or offline
      - if it is offline you MUST provide paths in the `model_type` param for each model under the abbr for the models
  - Optional Params
    - `bert`: 
      - default `model_type`: bert-base-uncased
    - `roberta`:
      - default `model_type`: robert-base
    - `xlm`:
      - default `model_type`: xlx-mlm-en-2048
    - `xlmr`:
      - default `model_type`: xlm-roberta-base
    - `xlnet`:
      - default `model_type`: xlnet-base-cased
    - Training args
      - under the abbvr for each model you can add these optional training args
      - `train_split`: float
        - Percentage of dataset dedicated to training 
        - Default: 0.8
      - `dev_split`: float
        - Percentage of dataset dedicated to training
        - Default: 0.1
    - `params`
      - all the hyperparameters in the deep learning optimization script have default values that you can override:
      - epochs
        - `epoch_min`: int,default = 1
        - `epoch_max`: int,default = 10
      - learning rate
        - `lr_min`: float,default = 1e-5
        - `lr_max`: float,default = 3e-4
      - weight decay (l2 norm)
        - `weight_decay_min`: float,default = 0.0
        - `weight_decay_max`: float,default = 0.05
      - gradient accumulation steps
        - `accumulation_steps`: List[int],default = [1,2,4]
      - learning rate scheduler
        - `scheduler`: List[str],default = ["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"]
      - learning rate warm up steps
        - `warmup_steps_min`: int,default = 100
        - `warmup_steps_max`: int,default = 1000
      - stochastic gradient descent batch size
        - `batch_size`: List[int],default = [4,8,16,32]
      - label smoothing in loss
        - `label_smoothing_min`: float,default = 0.0
        - `label_smoothing_max`: float,default = 0.1
      - multilayer perceptron size division rate
        - `mlp_division_min`: int,default = 2
        - `mlp_division_max`: int,default = 8
      - multilayer perceptron dropout 
        - `mlp_dropout_min`: float,default = 0.0
        - `mlp_dropout_max`: float,default = 0.5
      - ways of combining multimodal features
        - `combine_feat_method`: List[str],default = ["text_only","concat","mlp_on_categorical_then_concat","individual_mlps_on_cat_and_numerical_feats_then_concat","mlp_on_concatenated_cat_and_numerical_feats_then_concat","attention_on_cat_and_numerical_feats","gating_on_cat_and_num_feats_then_sum","weighted_feature_sum_on_transformer_cat_and_numerical_feats"]
      - gating strength for gating multimodal methods
        - `gating_beta_min`: float,default = 0.0
        - `gating_beta_max`: float,default = 0.4
      - method to ensemble models
        - `ensemble_method`: List[str],default = ["singular","bagging","fastgeometric","fusion","gradient","snapshot","softgradient","voting"]
      - number of estimators for ensemble
        - `n_estimators_min`: int,default = 2
        - `n_estimators_max`: int,default = 5
      - epochs for snapshot ensemble
        - `epochs_snapshot_min`: int,default = 1
        - `epochs_snapshot_max`: int,default = 4
  
## Generative
  - Required if stage is provided in pipeline setup
  - **IMPORTANT**
    - If you are using HuggingFace models using the API or using OpenAI models you must add their api keys to env variables
      - HF_ACCESS_TOKEN 
      - OPENAI_API_KEY
    - we have a `.env` file that will help with that
  - Required Params
    - `prompt`: str
      - Custom prompt that takes in column names into format string specification areas
      - For example if the dataframe has "text" and "name" as columns the corresponding prompt using them is: "{text} {name}"
    - `models` : List[str]/str
      - List of model types
      - Supported model types: "openai", "openaichat", "hfoffline", "hfonline"
      - Supported keywords: "all"
      - If you include one, you **MUST** have their required params listed as well
    - `label_set` : Dict[str, Any]
      - Mapping from responses from the model to the label in the dataframe prediction column
      - For example if you want yes no responses from the model that line up with true false in your dataframe it would look like
      ```yaml
        label_set:
        "yes" : true
        "no" : false
      ```
  - Optional Params
    - `hfoffline`
      - Required params
        - `model_names`: List[str]/str
          - List of huggingface models to use in offline context 
          - If downloaded, use the path to models instead
        - `batch_size`: int
          - size of the batch to feed to model
      - Optional params
        - All generation params for a model listed [here](https://huggingface.co/docs/transformers/main_classes/text_generation)
    - `hfonline`
      - Required params
        - `model_names`: List[str]/str
          - List of huggingface to use in online context
      - Optional params
        - All generation params for a model listed [here](https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task)
    - `openai`
      - Required params
        - `model_names`: List[str]/str
          - List of openai completion models (not the ones used for chatting) to use in online context
        - `batch_size`: int
          - size of batch to feed model
      - Optional params
        - All generation params for a model listed [here](https://platform.openai.com/docs/api-reference/completions/create)
    - `openaichat`
      - Required params
        - `model_names`: List[str]/str
          - List of openai chat models (not the ones used for completion) to use in online context
        - `messages`: Dict[str, Any]
          - Extra tidbits to prime the model for the specific task like role and starting message
          - Must have a under this `role` specified from `system`, `user`, `assistant`, or `function`
          - Optional params for this can be found [here](https://platform.openai.com/docs/api-reference/chat/create)
      - Optional params
        - All generation parms for a model listed [here](https://platform.openai.com/docs/api-reference/chat/create)
        
