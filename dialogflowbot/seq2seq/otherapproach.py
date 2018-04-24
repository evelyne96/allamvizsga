import data_work
import numpy
from numpy import argmax
from keras import optimizers
from keras.utils.vis_utils import plot_model
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed, Input, Bidirectional,Concatenate
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from random import randint
from numpy import array

source_ids, target_ids, vectorizer = data_work.read_preprocessed_data()
#id_to_word
inverse_vocabulary = {}
for (k,v) in enumerate(vectorizer.vocabulary_):
    inverse_vocabulary[k] = v

in_len = len(source_ids)
out_len = len(target_ids)

max_source_length = data_work.get_max_length(source_ids)+1
max_target_length = data_work.get_max_length(target_ids)+1

vocab_size = len(vectorizer.vocabulary_)
print('sources' ,in_len)
print('vocab', vocab_size)


unk_id = vocab_size   # 1
pad_id = vocab_size+1     # 0
start_id = vocab_size+2
stop_id = vocab_size+3
vocab_size += 4

train_source = data_work.pad_sentences(source_ids, pad_id, max_source_length)

target_to_decode = data_work.pad_sentences(data_work.append_stop(target_ids, stop_id), pad_id, max_target_length)

train_target = data_work.pad_sentences(data_work.append_start(target_ids, start_id), pad_id, max_target_length)

# The output sequence needs to be one-hot encoded. 
# This is because the model will predict the probability of each word in the vocabulary as output.
target_to_decode = data_work.encode_output(target_to_decode, vocab_size)

def define_model(src_vocab_size, trg_vocab_size, n_unit, bi_directional, dropout = 0.2):
    encoder_inputs = Input(shape=(None,))
    emb_inp = Embedding(src_vocab_size, n_unit, mask_zero=True)(encoder_inputs)

    if bi_directional:
        encoder_lstm = Bidirectional(LSTM(n_unit, return_state=True,dropout = dropout))
        dec_unit = n_unit * 2

        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(emb_inp)
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]
    else:
        encoder_lstm = LSTM(n_unit, return_state=True)
        dec_unit = n_unit
        encoder_outputs, state_h, state_c = encoder_lstm(emb_inp)
        encoder_states = [state_h, state_c]

    # define training decoder
    decoder_inputs = Input(shape=(None, ))
    emb2 = Embedding(trg_vocab_size, n_unit)
    emp_inp_trg = emb2(decoder_inputs)
    decoder_lstm = LSTM(dec_unit, return_sequences=True, return_state=True,dropout = dropout)
    decoder_outputs, _, _ = decoder_lstm(emp_inp_trg, initial_state=encoder_states)
    decoder_dense = Dense(trg_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(dec_unit,))
    decoder_state_input_c = Input(shape=(dec_unit,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    emb_inf = emb2(decoder_inputs)
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(emb_inf, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

    # return all models
    return model, encoder_model, decoder_model


def train_model(model, m2 ,m3 , train_in, train_out, target_to_decode):
    # filename = 'train_model.h5'
    # checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # model.fit([train_in, train_out], target_to_decode, epochs=30, batch_size=256, validation_split=0.1,\
            # callbacks=[checkpoint], verbose=1)
    n = 0
    while True:
        print('epoch: ',n)
        model.fit([train_in, train_out], target_to_decode, epochs=1, batch_size=256, validation_split=0.1, verbose=1)
        n += 1
        if n % 10 == 0:
            save_models_to_file(model, m2, m3)
            # to comply with the recommendations
            test = 'to implement the recommendations'
            decode_sequence(test, m2, m3, vocab_size, max_target_length)

def decode_sequence(input_seq, encoder_model, decoder_model, num_decoder_tokens, max_len):
    # Encode the input as state vectors.
    s = list()
    s.append(data_work.get_ids_for_sentence(data_work.word_tokenize(input_seq), vectorizer.vocabulary_)[:30])
    src = data_work.pad_sentences(s, pad_id, max_source_length)
    states_value = encoder_model.predict(src)

    # Generate empty target sequence of length 1.
    target_seq = numpy.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0,0] = start_id
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # Sample a token

        sampled_token_index = argmax(output_tokens[0, -1, :])
        # Exit condition: either hit max length
        # or find stop character.
        if ((len(decoded_sentence)+1) == max_len):
            stop_condition = True

        try:
            sampled_word = inverse_vocabulary[sampled_token_index]
            decoded_sentence.append(sampled_word)
        except:
            break

        # Update the target sequence (of length 1).
        target_seq = numpy.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        # Update states
        states_value = [h, c]

    print(decoded_sentence)
    return decoded_sentence

def save_models_to_file(m1, m2, m3):
    m1.save('models/train_model.h5')
    m2.save('models/encoder_model.h5')
    m3.save('models/decoder_model.h5')

def load_models_from_files():
    train_m = load_model('models/train_model.h5')    
    # print(train_m.summary())
    decoder_model = load_model('models/decoder_model.h5')
    # print(decoder_model.summary())
    encoder_model = load_model('models/encoder_model.h5')
    # print(encoder_model.summary())
    return train_m, encoder_model, decoder_model


model, encoder_model, decoder_model = define_model(vocab_size, vocab_size, 256, True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())
train_model(model,encoder_model, decoder_model, train_source, train_target, target_to_decode)

# model, encoder_model, decoder_model = load_models_from_files()
# test = 'proposed by the president of the'
# decode_sequence(test, encoder_model, decoder_model, vocab_size, max_target_length)












#for debugging purposes
def decode_sequence2(input_seq, encoder_model, decoder_model, num_decoder_tokens, max_len):
    # Encode the input as state vectors.
    x = array(input_seq)
    states_value = encoder_model.predict(x)

    target_seq = numpy.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0,0] = start_id
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # Sample a token
        sampled_token_index = argmax(output_tokens[0, 0, :])
        decoded_sentence.append(sampled_token_index)
        print(sampled_token_index)
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_token_index == stop_id  or (len(decoded_sentence)) == max_len):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = numpy.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    print(decoded_sentence)
    return decoded_sentence

def generate_sequence(length, n_unique):
	return [randint(1, n_unique-1) for _ in range(length)]

def get_dataset(n_in, n_out, cardinality, n_samples):
    X1, X2, y = list(), list(), list()
    for _ in range(n_samples):
		# generate source sequence
        source = generate_sequence(n_in, cardinality)
		# define target sequence
        target = source[:n_out]
        target.reverse()
        target.append(stop_id)
		# create padded input target sequence
        target_in = [start_id] + target[:-1]
		# encode
		# src_encoded = data_work.to_categorical([source], num_classes=cardinality)
		# tar2_encoded = data_work.to_categorical([target_in], num_classes=cardinality)
		# store
        X1.append(source)
        X2.append(target_in)
        y.append(target)

    y =  data_work.encode_output(array(y), cardinality)
    return array(X1), array(X2), array(y)


# stop_id = 101
# start_id = 0
# 3+1 tehat 3 hosszu eredmeny
# train_source,train_target, target_to_decode = get_dataset(8, 3, 102, 10000)
# model, encoder_model, decoder_model = define_model(102, 102, 256, True)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# # print(model.summary())
# train_model(model,encoder_model, decoder_model, train_source, train_target, target_to_decode)

# model, encoder_model, decoder_model = load_models_from_files()

# test = [[88,77,66,55,44, 33, 22, 11]]
# decode_sequence2(test, encoder_model, decoder_model, 102, 3)