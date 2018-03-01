from __future__ import absolute_import, division, print_function

import os
import pickle
from six.moves import urllib

import tflearn
from tflearn.data_utils import *

# Config
loading_from_checkpoint = False
# checkpoint_model_path = './quijote_trained_model/model_quijote-275408'
num_epochs = 50

path = "donquijote.txt"
char_idx_file = 'char_idx.pickle'

# if not os.path.isfile(path):
#     urllib.request.urlretrieve("https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/shakespeare_input.txt", path)

maxlen = 25

char_idx = None
if os.path.isfile(char_idx_file):
  print('Loading previous char_idx')
  char_idx = pickle.load(open(char_idx_file, 'rb'))

X, Y, char_idx = \
    textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3,
                                         pre_defined_char_idx=char_idx)

pickle.dump(char_idx, open(char_idx_file,'wb'))

g = tflearn.input_data([None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 1024, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 1024, return_seq=True)
g = tflearn.dropout(g, 0.8)
g = tflearn.lstm(g, 1024, return_seq=True)
g = tflearn.dropout(g, 0.85)
g = tflearn.lstm(g, 1024)
g = tflearn.dropout(g, 0.85)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.001)

m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='model_quijote')

if loading_from_checkpoint == True:
  m.load(checkpoint_model_path)

for i in range(num_epochs):
    seed = random_sequence_from_textfile(path, maxlen)
    m.fit(X, Y, validation_set=0.1, batch_size=128,
          n_epoch=1, run_id='quijote-compatible-yet-beefy')
    print("-- TESTING...")
    print("-- Test with temperature of 1.0 --")
    print(m.generate(600, temperature=1.0, seq_seed=seed))
    print("-- Test with temperature of 0.5 --")
    print(m.generate(600, temperature=0.5, seq_seed=seed))
