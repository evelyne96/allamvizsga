# https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/
import data_work
from numpy import argmax
from keras.utils.vis_utils import plot_model
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.layers import Input


source_ids, target_ids, vectorizer = data_work.read_preprocessed_data()
#id_to_word
inverse_vocabulary = {}
for (k,v) in enumerate(vectorizer.vocabulary_):
    inverse_vocabulary[k] = v

in_len = len(source_ids)
out_len = len(target_ids)

max_source_length = data_work.get_max_length(source_ids)
max_target_length = data_work.get_max_length(target_ids)

vocab_size = len(vectorizer.vocabulary_)


unk_id = vocab_size   # 1
pad_id = vocab_size+1     # 0
vocab_size += 2


train_source = data_work.pad_sentences(source_ids, pad_id, max_source_length)
train_source_test = train_source[3700:]
train_source = train_source[:3700]
train_target = data_work.pad_sentences(target_ids, pad_id, max_target_length)
train_target_test = train_target[3700:]
train_target = train_target[:3700]
# The output sequence needs to be one-hot encoded. 
# This is because the model will predict the probability of each word in the vocabulary as output.
train_target = data_work.encode_output(train_target, vocab_size)
train_target_test = data_work.encode_output(train_target_test, vocab_size)

# encoder-decoder LSTM
def define_model(src_vocab_size, trg_vocab_size, in_len, out_len, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab_size, n_units, input_length=in_len, mask_zero=True))
    model.add(LSTM(n_units))
    # To summarize, the RepeatVector is used as an adapter to fit the fixed-sized 2D output of the encoder to the differing 
    # length and 3D input expected by the decoder.
    # The TimeDistributed wrapper allows the same output layer to be reused for each element in the output sequence.
    model.add(RepeatVector(out_len))
    model.add(LSTM(n_units, return_sequences=True))
    model.add(TimeDistributed(Dense(trg_vocab_size, activation='softmax')))
    return model

def train_model(model, train_in, train_out, train_in_test, train_out_test):
    filename = 'model.h5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit(train_in, train_out, epochs=100, batch_size=64, validation_data=(train_in_test, train_out_test),\
            callbacks=[checkpoint], verbose=2)


def predict_sequence(model, source):
    s = []
    s.append(data_work.get_ids_for_sentence(data_work.word_tokenize(source), vectorizer.vocabulary_)[:30])
    src = data_work.pad_sentences(s, pad_id, max_source_length)
    paraphrase = model.predict(src, verbose=1)[0]
    # decode a one hot encoded string
    integers = [argmax(vector) for vector in paraphrase]
    target = list()
    for i in integers:
        try:
            word = inverse_vocabulary[i]
        except:
            word = 'unk'
        target.append(word)
    return ' '.join(target)

#model for training
# rms = RMSprop() #another optimazer
model = define_model(vocab_size, vocab_size, max_source_length, max_target_length, 256)
model.compile(optimizer='adam', loss='categorical_crossentropy')
# print(model.summary())
train_model(model, train_source, train_target, train_source_test, train_target_test)

# test = 'TThe settling companies would also assign their possible claims against the underwriters to the investor plaintiffs, he added.	Under the agreement, the settling companies will also assign their potential claims against the underwriters to the investors, he added.'
# model = load_model('model.h5')
# print(predict_sequence(model, test))



# def define_models(n_input, n_output, n_units):
# define training encoder
# 	encoder_inputs = Input(shape=(None, n_input))
# 	encoder = LSTM(n_units, return_state=True)
# 	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# 	encoder_states = [state_h, state_c]
# 	# define training decoder
# 	decoder_inputs = Input(shape=(None, n_output))
# 	decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
# 	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
# 	decoder_dense = Dense(n_output, activation='softmax')
# 	decoder_outputs = decoder_dense(decoder_outputs)
# 	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# 	# define inference encoder
# 	encoder_model = Model(encoder_inputs, encoder_states)
# 	# define inference decoder
# 	decoder_state_input_h = Input(shape=(n_units,))
# 	decoder_state_input_c = Input(shape=(n_units,))
# 	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# 	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
# 	decoder_states = [state_h, state_c]
# 	decoder_outputs = decoder_dense(decoder_outputs)
# 	decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
# 	# return all models
# 	return model, encoder_model, decoder_model

# define_models(max_source_length, max_target_length, 124)





