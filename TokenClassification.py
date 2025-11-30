# Named Entity Recognition with Pre-Trained Model
# Download and Preprocess Data
import os
import wget

# set data path
DATA_DIR="data/GMB"

# check that data folder should contain 4 files
!ls -l $DATA_DIR

# preview data 
print('Text:')
!head -n 5 {DATA_DIR}/text_train.txt

print('Labels:')
!head -n 5 {DATA_DIR}/labels_train.txt

# Labeling Data
https://datasaur.ai/sign-up/
https://www.youtube.com/watch?v=I9WVmnnSciE

# Use Pre-Trained Model
# Download Pre-Trained Model
# import dependencies
from nemo.collections.nlp.models import TokenClassificationModel

# list available pre-trained models
for model in TokenClassificationModel.list_available_models():
    print(model)

# download and load the pre-trained BERT-based model
pretrained_ner_model=TokenClassificationModel.from_pretrained("ner_en_bert")


# Make Predictions
# define the list of queries for inference
queries=[
    'we bought four shirts from the nvidia gear store in santa clara.',
    'Nvidia is a company.',
]

# make sample predictions
results=pretrained_ner_model.add_predictions(queries)

# show predictions
for query, result in zip(queries, results):
    print(f'Query : {query}')
    print(f'Result: {result.strip()}\n')
    print()

# Evaluate Predictions
# create a subset of our dev data
!head -n 100 $DATA_DIR/text_dev.txt > $DATA_DIR/sample_text_dev.txt
!head -n 100 $DATA_DIR/labels_dev.txt > $DATA_DIR/sample_labels_dev.txt

WORK_DIR = "WORK_DIR"

# evaluate model performance on sample
pretrained_ner_model.evaluate_from_file(
    text_file=os.path.join(DATA_DIR, 'sample_text_dev.txt'),
    labels_file=os.path.join(DATA_DIR, 'sample_labels_dev.txt'),
    output_dir=WORK_DIR,
    add_confusion_matrix=True,
    normalize_confusion_matrix=True,
    batch_size=1
)

# Fine-Tune a Pre-Trained Model
# the dataset directory is the only required argument if the files names are labels_dev.txt, labels_train.txt, text_dev.txt, and text_train.txt.
import pytorch_lightning as pl

# setup the data dir to get class weights statistics
pretrained_ner_model.update_data_dir(DATA_DIR)

# setup train and validation Pytorch DataLoaders
pretrained_ner_model.setup_training_data()
pretrained_ner_model.setup_validation_data()

# set up loss
pretrained_ner_model.setup_loss()

# create a PyTorch Lightning trainer and call `fit` again
fast_dev_run=True
trainer=pl.Trainer(devices=1, accelerator='gpu', fast_dev_run=fast_dev_run)
trainer.fit(pretrained_ner_model)

# evaluate model performance on sample
pretrained_ner_model.evaluate_from_file(
    text_file=os.path.join(DATA_DIR, 'sample_text_dev.txt'),
    labels_file=os.path.join(DATA_DIR, 'sample_labels_dev.txt'),
    output_dir=WORK_DIR,
    add_confusion_matrix=True,
    normalize_confusion_matrix=True,
    batch_size=1
)

