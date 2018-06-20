import re, csv, os
from numpy import array
from tensorflow.python.platform import gfile
from nltk.tokenize import sent_tokenize, word_tokenize
from random import shuffle
from data_work import read_quora_data, read_pdb_data, read_msr_data

train_data_path = os.path.join(os.getcwd(),'data','MSRParaphraseCorpus', 'msr_paraphrase_train.txt')
train_data_path2 = os.path.join(os.getcwd(),'data','MSRParaphraseCorpus', 'quora_duplicate_questions.tsv')
train_data_path3 = os.path.join(os.getcwd(),'data','MSRParaphraseCorpus', 'data-set.txt')

def read_data():
    content =[]
    sentences = []
    content, sentences = read_pdb_data(content, sentences)
    shuffle(content)
    content = content[:70000]
    content, sentences = read_quora_data(content, sentences)
    # content, sentences = read_msr_data(content, sentences)
    return content, sentences

content, _ = read_data()
shuffle(content)
print(len(content))

f = open('data/nmt-keras-data/training.trg', 'w')
f.close()

f = open('data/nmt-keras-data/training.src', 'w')
f.close()

f = open('data/nmt-keras-data/dev.src', 'w')
f.close()

f = open('data/nmt-keras-data/dev.trg', 'w')
f.close()

f = open('data/nmt-keras-data/test.trg', 'w')
f.close()

f = open('data/nmt-keras-data/test.src', 'w')
f.close()

train_trg = open('data/nmt-keras-data/training.trg','a', encoding="utf-8") 
train_src = open('data/nmt-keras-data/training.src','a', encoding="utf-8") 
dev_src = open('data/nmt-keras-data/dev.src','a', encoding="utf-8")
dev_trg = open('data/nmt-keras-data/dev.trg','a', encoding="utf-8")
test_trg = open('data/nmt-keras-data/test.trg','a', encoding="utf-8")
test_src = open('data/nmt-keras-data/test.src','a', encoding="utf-8")


from random import shuffle
shuffle(content)
train_data = content[:175000]
test_data = content[175001:193000]
dev_data = content[193001:]


for (src,trg) in train_data:
    train_src.write(src)
    train_src.write('\n')
    train_trg.write(trg)

train_src.close()
train_trg.close()

for (src,trg) in test_data:
    test_src.write(src)
    test_src.write('\n')
    test_trg.write(trg)

test_src.close()
test_trg.close()

for (src,trg) in dev_data:
    # print(src,"000",trg)
    dev_src.write(src)
    dev_src.write('\n')
    dev_trg.write(trg)

dev_src.close()
dev_trg.close()





