# Download and Preprocess Data
import os
import wget

# set data path
DATA_DIR='data'
DATA_DIR=os.path.join(DATA_DIR, 'SST-2')

# check that data folder should contain train.tsv, dev.tsv, test.tsv
!ls -l {DATA_DIR}

# preview data 
print('Train:')
!head -n 5 {DATA_DIR}/train.tsv

print('Dev:')
!head -n 5 {DATA_DIR}/dev.tsv

print('Test:')
!head -n 5 {DATA_DIR}/test.tsv

!sed 1d {DATA_DIR}/train.tsv > {DATA_DIR}/train_nemo_format.tsv
!sed 1d {DATA_DIR}/dev.tsv > {DATA_DIR}/dev_nemo_format.tsv

# Fine-Tune a Pre-Trained Model
# define config path
# Configuration File
MODEL_CONFIG="text_classification_config.yaml"
WORK_DIR='WORK_DIR'
os.makedirs(WORK_DIR, exist_ok=True)

# download the model's configuration file 
BRANCH='main'
config_dir = WORK_DIR + '/configs/'
os.makedirs(config_dir, exist_ok=True)
if not os.path.exists(config_dir + MODEL_CONFIG):
    print('Downloading config file...')
    wget.download(f'https://raw.githubusercontent.com/NVIDIA/NeMo/{BRANCH}/examples/nlp/text_classification/conf/' + MODEL_CONFIG, config_dir)
else:
    print ('config file already exists')

from omegaconf import OmegaConf

CONFIG_DIR = "/dli/task/WORK_DIR/configs"
CONFIG_FILE = "text_classification_config.yaml"

config=OmegaConf.load(CONFIG_DIR + "/" + CONFIG_FILE)

# print the entire configuration file
print(OmegaConf.to_yaml(config))

!ls $DATA_DIR

# set num_classes to 2
config.model.dataset.num_classes=2

# set file paths
config.model.train_ds.file_path = os.path.join(DATA_DIR, 'train_nemo_format.tsv')
config.model.validation_ds.file_path = os.path.join(DATA_DIR, 'dev_nemo_format.tsv')

# You may change other params like batch size or the number of samples to be considered (-1 means all the samples)

# print the model section
print(OmegaConf.to_yaml(config.model))

# lets modify some trainer configs

# setup max number of steps to reduce training time for demonstration purposes of this tutorial
# Training stops when max_step or max_epochs is reached (earliest)
config.trainer.max_epochs = 1

# print the trainer section
print(OmegaConf.to_yaml(config.trainer))

# Download Pre-Trained Model
from nemo.collections import nlp as nemo_nlp

# complete list of supported BERT-like models
for model in nemo_nlp.modules.get_pretrained_lm_models_list(): 
    print(model)

# specify the BERT-like model, you want to use
# set the `model.language_modelpretrained_model_name' parameter in the config to the model you want to use
config.model.language_model.pretrained_model_name = "bert-base-uncased"

from nemo.collections.nlp.models import TextClassificationModel
import pytorch_lightning as pl

trainer=pl.Trainer(**config.trainer)
text_classification_model=TextClassificationModel(cfg=config.model, trainer=trainer)

# start model training
trainer.fit(text_classification_model)

eval_config = OmegaConf.create({'file_path': config.model.validation_ds.file_path, 'batch_size': 64, 'shuffle': False, 'num_samples': -1})
text_classification_model.setup_test_data(test_data_config=eval_config)
trainer.test(model=text_classification_model, verbose=False)

# define the list of queries for inference
queries = ['by the end of no such thing the audience , like beatrice , has a watchful affection for the monster .', 
           'director rob marshall went out gunning to make a great one .', 
           'uneasy mishmash of styles and genres .']
           
# max_seq_length=512 is the maximum length BERT supports.       
results = text_classification_model.classifytext(queries=queries, batch_size=3, max_seq_length=512)

print('The prediction results of some sample queries with the trained model:')
for query, result in zip(queries, results):
    print(f'Query : {query}')
    print(f'Predicted label: {result}')


