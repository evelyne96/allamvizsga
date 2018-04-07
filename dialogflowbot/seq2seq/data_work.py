import csv, os
from numpy import array
from tensorflow.python.platform import gfile
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

train_data_path = os.path.join(os.getcwd(),'data','MSRParaphraseCorpus', 'msr_paraphrase_train.txt')
train_pairs_path = os.path.join(os.getcwd(),'data','preprocessed', 'training_pairs.pkl')
train_pairs_ids_path = os.path.join(os.getcwd(),'data','preprocessed', 'training_pairs_ids.pkl')
train_vectorizer = os.path.join(os.getcwd(),'data','preprocessed', 'train_vectorizer.pkl')
go = 0
def trunc_at(s, d, n=3):
    "Returns s truncated at the n'th (3rd by default) occurrence of the delimiter, d."
    return d.join(s.split(d, n)[n:])

def read_data(data_path):
    content =[]
    sentences = []
    if gfile.Exists(data_path):
        with gfile.GFile(data_path, mode="rb") as dfile:
            for line in dfile:
                try:
                    line = trunc_at(line.decode('utf-8'), '\t') 
                    l = sent_tokenize(line)
                    if len(l) == 2:
                        # sentence - paraphrase pairs \t start \n end
                        content.append((l[0],l[1]))
                        sentences.append(l[0])
                        sentences.append(l[1])
                except:
                    print("exception when tokenizing or decoding")
    return content, sentences

def read_preprocessed_data():
    train_pairs = None
    vectorizer = None
    with (open(train_pairs_ids_path, "rb")) as openfile:
        train_pairs = pickle.load(openfile)
    with (open(train_vectorizer, "rb")) as openfile:
        vectorizer = pickle.load(openfile)
    source = [x for (x,y) in train_pairs]
    target = [x for (y,x) in train_pairs]
    return source, target, vectorizer
    

def get_sentence_token_ids(pairs, vocabulary):
    token_pairs = []
    for s1, s2 in pairs:
        s1 = s1.replace('-'," ")
        token_pairs.append((get_ids_for_sentence(word_tokenize(s1), vocabulary), \
                             get_ids_for_sentence(word_tokenize(s2), vocabulary)))
    return token_pairs


def get_ids_for_sentence(s, vocabulary):
    s_ids = []
    for w in s:
            try:
                s_ids.append(vocabulary[w.lower()])
            except:
                x=0
    return s_ids

def save_preprocessed(pairs, vectorizer, token_id_pairs):
    with open(train_pairs_path, 'wb') as f:
          pickle.dump(pairs, f)
    with open(train_pairs_ids_path, 'wb') as f:
          pickle.dump(token_id_pairs, f)
    with open(train_vectorizer, 'wb') as f:
          pickle.dump(vectorizer, f)


def get_vocab():
    pairs, sentences = read_data(train_data_path)
    c_vectorizer = CountVectorizer()
    features = c_vectorizer.fit_transform(sentences)
    token_id_pairs = get_sentence_token_ids(pairs, c_vectorizer.vocabulary_)
    # save_preprocessed(pairs, c_vectorizer, token_id_pairs)

def get_max_length(data):
    return max(len(d) for d in data)

def pad_sentences(data, pad_id, length):
    data = pad_sequences(data, maxlen=length, padding='post', value=pad_id)
    return data

def encode_output(data, vocab_size):
    encoded_data = []
    ecoded = []
    for d in data:
        encoded = to_categorical(d, num_classes=vocab_size)
        encoded_data.append(encoded)
    encoded_data = array(encoded_data)
    encoded_data = encoded_data.reshape(data.shape[0], data.shape[1], vocab_size)
    return encoded_data

get_vocab()