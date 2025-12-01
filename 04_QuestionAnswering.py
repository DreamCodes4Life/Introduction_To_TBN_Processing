# Extractive Question-Answering with BERT-like models
BRANCH = 'main'

# Imports and constants
import os
import wget
import gc

import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.nlp.models.question_answering.qa_bert_model import BERTQAModel

gc.disable()

# set the following paths
DATA_DIR = "data" # directory for storing datasets
WORK_DIR = "work_dir" # directory for storing trained models, logs, additionally downloaded scripts

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

# download the model's default configuration file 
config_dir = WORK_DIR + '/conf/'
os.makedirs(config_dir, exist_ok=True)
if not os.path.exists(config_dir + "qa_conf.yaml"):
    print('Downloading config file...')
    wget.download(f'https://raw.githubusercontent.com/NVIDIA/NeMo/{BRANCH}/examples/nlp/question_answering/conf/qa_conf.yaml', config_dir)
else:
    print ('config file already exists')

# this will print the entire default config of the model
config_path = f'{WORK_DIR}/conf/qa_conf.yaml'
print(config_path)
config = OmegaConf.load(config_path)
print("Default Config - \n")
print(OmegaConf.to_yaml(config))

# Training and testing models on SQuAD v2.0
!ls -LR {DATA_DIR}/squad

# Set dataset config values
# if True, model will load features from cache if file is present, or
# create features and dump to cache file if not already present
config.model.dataset.use_cache = False

# indicates whether the dataset has unanswerable questions
config.model.dataset.version_2_with_negative = True

# indicates whether the dataset is of extractive nature or not
# if True, context spans/chunks that do not contain answer are treated as unanswerable 
config.model.dataset.check_if_answer_in_context = True

# set file paths for train, validation, and test datasets
config.model.train_ds.file = f"{DATA_DIR}/squad/v2.0/train-v2.0.json"
config.model.validation_ds.file = f"{DATA_DIR}/squad/v2.0/dev-v2.0.json"
config.model.test_ds.file = f"{DATA_DIR}/squad/v2.0/dev-v2.0.json"

# set batch sizes for train, validation, and test datasets
config.model.train_ds.batch_size = 8
config.model.validation_ds.batch_size = 8
config.model.test_ds.batch_size = 8

# set number of samples to be used from dataset. setting to -1 uses entire dataset
config.model.train_ds.num_samples = 5000
config.model.validation_ds.num_samples = 1000
config.model.test_ds.num_samples = 100

# Set trainer config values
config.trainer.max_epochs = 1
config.trainer.max_steps = -1 # takes precedence over max_epochs
config.trainer.precision = 16
config.trainer.devices = [0] # 0 for CPU, or list of the GPUs to use [0] this tutorial does not support multiple GPUs. If needed please use NeMo/examples/nlp/question_answering/question_answering.py
config.trainer.accelerator = "gpu"
config.trainer.strategy="auto"

# BERT model for SQuAD v2.0
# Set model config values
# set language model and tokenizer to be used
# tokenizer is derived from model if a tokenizer name is not provided
config.model.language_model.pretrained_model_name = "bert-base-uncased"
config.model.tokenizer.tokenizer_name = "bert-base-uncased"

# path where model will be saved
config.model.nemo_path = f"{WORK_DIR}/checkpoints/bert_squad_v2_0.nemo"

# config.exp_manager.create_checkpoint_callback = True

config.model.optim.lr = 3e-5

# Create trainer and initialize model
trainer = pl.Trainer(**config.trainer)
model = BERTQAModel(config.model, trainer=trainer)

# Train, test, and save the model
trainer.fit(model)
trainer.test(model)

model.save_to(config.model.nemo_path)

# Load the saved model and run inference
model = BERTQAModel.restore_from(config.model.nemo_path)

eval_device = [config.trainer.devices[0]] if isinstance(config.trainer.devices, list) else 1
model.trainer = pl.Trainer(
    devices=eval_device,
    accelerator=config.trainer.accelerator,
    precision=16,
    logger=False,
)

all_preds, all_nbest = model.inference(
    config.model.test_ds.file,
    num_samples=10, # setting to -1 will use all samples for inference
)

for question_id in all_preds:
    print(all_preds[question_id])
