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

