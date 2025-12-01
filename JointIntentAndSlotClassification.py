BRANCH = 'main'

from nemo.collections import nlp as nemo_nlp
from nemo.utils import logging

import os
import wget
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

# you can replace DATA_DIR and NEMO_DIR with your own locations
DATA_DIR = "data"
DATA_DIR=os.path.join(DATA_DIR, 'NLU-Evaluation-Data')

# # download and unzip the example dataset from github
TEST_DIR='.'
print('Downloading dataset...')
wget.download('https://github.com/xliuhw/NLU-Evaluation-Data/archive/master.zip', TEST_DIR)
!unzip {TEST_DIR}/NLU-Evaluation-Data-master.zip -d {TEST_DIR}

# convert the dataset to the NeMo format
!python import_datasets.py --dataset_name=assistant --source_data_dir={DATA_DIR} 

# list of queries divided by intent files in the original training dataset
! ls -l {DATA_DIR}/dataset/trainset

# print all intents from the NeMo format intent dictionary
!echo 'Intents: ' $(wc -l < {DATA_DIR}/nemo_format/dict.intents.csv)
!cat {DATA_DIR}/nemo_format/dict.intents.csv

# print all slots from the NeMo format slot dictionary
!echo 'Slots: ' $(wc -l < {DATA_DIR}/nemo_format/dict.slots.csv)
!cat {DATA_DIR}/nemo_format/dict.slots.csv

# examples from the intent training file
!head -n 10 {DATA_DIR}/nemo_format/train.tsv

# examples from the slot training file
!head -n 10 {DATA_DIR}/nemo_format/train_slots.tsv

WORK_DIR='WORK_DIR'
MODEL_CONFIG = "intent_slot_classification_config.yaml"
config_dir = WORK_DIR + '/configs/'
os.makedirs(config_dir, exist_ok=True)
if not os.path.exists(config_dir + MODEL_CONFIG):
    print('Downloading config file...')
    wget.download(f'https://raw.githubusercontent.com/NVIDIA/NeMo/{BRANCH}/examples/nlp/intent_slot_classification/conf/' + MODEL_CONFIG, config_dir)
else:
    print ('config file already exists')

config_file = "intent_slot_classification_config.yaml"
config=OmegaConf.load(config_dir + "/" + config_file)

# print the entire configuration file
print(OmegaConf.to_yaml(config))

config.model.data_dir = f'{DATA_DIR}/nemo_format'

# Building the PyTorch Lightning Trainer
# lets modify some trainer configs
# checks if we have GPU available and uses it
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
config.trainer.devices = 1
config.trainer.accelerator = accelerator

config.trainer.precision = 16 if torch.cuda.is_available() else 32

# for mixed precision training, uncomment the line below (precision should be set to 16 and amp_level to O1):
# config.trainer.amp_level = O1

# remove distributed training flags
config.trainer.strategy = 'auto'

# setup a small number of epochs for demonstration purposes of this tutorial
config.trainer.max_epochs = 5

trainer = pl.Trainer(**config.trainer)

# Initializing the model and Training
# initialize the model
model = nemo_nlp.models.IntentSlotClassificationModel(config.model, trainer=trainer)

# train
trainer.fit(model)

# we will setup testing data reusing the same config (test section)
model.setup_test_data(test_data_config=config.model.test_ds)

# run the evaluation on the test dataset
trainer.test(model=model, verbose=False)

queries = [
    'set alarm for seven thirty am',
    'lower volume by fifty percent',
    'what is my schedule for tomorrow',
]

# pred_intents, pred_slots = eval_model.predict_from_examples(queries, config.model.test_ds)
pred_intents, pred_slots = model.predict_from_examples(queries, config.model.test_ds)

logging.info('The prediction results of some sample queries with the trained model:')
for query, intent, slots in zip(queries, pred_intents, pred_slots):
    logging.info(f'Query : {query}')
    logging.info(f'Predicted Intent: {intent}')
    logging.info(f'Predicted Slots: {slots}')

