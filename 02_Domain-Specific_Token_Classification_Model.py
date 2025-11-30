# Download Data
import os
import wget

# set data path
DATA_DIR = "data/NCBI"
os.makedirs(DATA_DIR, exist_ok=True)

# example
with open(f'{DATA_DIR}/NCBI_corpus_testing.txt') as f: 
    sample_text=f.readline()
    
print(sample_text)

# we see the following tags
import re

# use regular expression to find labels
categories=re.findall('<category.*?<\/category>', sample_text)
for sample in categories: 
    print(sample)


# For NER, the dataset labels individual words as diseases.
NER_DATA_DIR = f'{DATA_DIR}/NER'
os.makedirs(os.path.join(DATA_DIR, 'NER'), exist_ok=True)

# show downloaded files
!ls -lh $NER_DATA_DIR

!head $NER_DATA_DIR/train.tsv

# Preprocess Data
# invoke the conversion script 
!python import_from_iob_format.py --data_file=$NER_DATA_DIR/train.tsv
!python import_from_iob_format.py --data_file=$NER_DATA_DIR/dev.tsv
!python import_from_iob_format.py --data_file=$NER_DATA_DIR/test.tsv

# preview dataset
!head -n 1 $NER_DATA_DIR/text_train.txt
!head -n 1 $NER_DATA_DIR/labels_train.txt

# Fine-Tune a Pre-Trained Model for Custom Domain
# Configuration File
# define config path
MODEL_CONFIG = "token_classification_config.yaml"
WORK_DIR = "WORK_DIR"
os.makedirs(WORK_DIR, exist_ok=True)

# download the model's configuration file 
BRANCH = 'main'
config_dir = WORK_DIR + '/configs/'
os.makedirs(config_dir, exist_ok=True)

if not os.path.exists(config_dir + MODEL_CONFIG):
    print('Downloading config file...')
    wget.download(f'https://raw.githubusercontent.com/NVIDIA/NeMo/{BRANCH}/examples/nlp/token_classification/conf/' + MODEL_CONFIG, config_dir)
else:
    print ('config file already exists')


from omegaconf import OmegaConf

CONFIG_DIR = "/dli/task/WORK_DIR/configs"
CONFIG_FILE = "token_classification_config.yaml"

config=OmegaConf.load(CONFIG_DIR + "/" + CONFIG_FILE)

# print the entire configuration file
print(OmegaConf.to_yaml(config))

# in this exercise, train and dev datasets are located in the same folder under the default names, 
# so it is enough to add the path of the data directory to the config
config.model.dataset.data_dir = os.path.join(DATA_DIR, 'NER')

# print the model section
print(OmegaConf.to_yaml(config.model))

# Download Domain-Specific Pre-Trained Model
# import dependencies
from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel

# list available pre-trained models
for model in MegatronBertModel.list_available_models(): 
    print(model.pretrained_model_name)

# To load the pretrained BERT LM model, we change the model.language_mode argument in the config as well as a few other arguments.
# add the specified above model parameters to the config
MODEL_NAME='biomegatron345m_biovocab_30k_cased'
# MODEL_NAME='biomegatron-bert-345m-cased'

config.model.language_model.lm_checkpoint=None
config.model.language_model.pretrained_model_name=MODEL_NAME
config.model.tokenizer.tokenizer_name=None

# use appropriate configurations based on GPU capacity
config.model.dataset_max_seq_length=64
config.model.train_ds.batch_size=32
config.model.validation_ds.batch_size=32
config.model.test_ds.batch_size=32

# limit the number of epochs for this demonstration
config.trainer.max_epochs=1
# config.trainer.precision=16
# config.trainer.amp_level='O1'

#  Instantiate Model and Trainer
# create trainer and model instances
from nemo.collections.nlp.models import TokenClassificationModel
import pytorch_lightning as pl

trainer=pl.Trainer(**config.trainer)
ner_model=TokenClassificationModel(cfg=config.model, trainer=trainer)

# start model training
trainer.fit(ner_model)

# Model Evaluation
# create a subset of our dev data
!head -n 100 $NER_DATA_DIR/text_dev.txt > $NER_DATA_DIR/sample_text_dev.txt
!head -n 100 $NER_DATA_DIR/labels_dev.txt > $NER_DATA_DIR/sample_labels_dev.txt

# evaluate model performance on sample
ner_model.half().evaluate_from_file(
    text_file=os.path.join(NER_DATA_DIR, 'sample_text_dev.txt'),
    labels_file=os.path.join(NER_DATA_DIR, 'sample_labels_dev.txt'),
    output_dir=WORK_DIR,
    add_confusion_matrix=True,
    normalize_confusion_matrix=True,
    batch_size=1
)
