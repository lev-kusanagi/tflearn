from __future__ import absolute_import, division, print_function

import os
import pickle
# from six.moves import urllib

import tflearn
from tflearn.data_utils import *

dir_path = os.path.dirname(os.path.realpath(__file__))
path = dir_path + "/donquijote.txt"
char_idx_file = dir_path + '/char_idx.pickle'
model_path = dir_path + "/quijote_trained_model/model_quijote-245900"

maxlen = 25

char_idx = None
if os.path.isfile(char_idx_file):
  char_idx = pickle.load(open(char_idx_file, 'rb'))

# X, Y, char_idx = \
#     textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3,
#                                          pre_defined_char_idx=char_idx)

# pickle.dump(char_idx, open(char_idx_file,'wb'))

g = tflearn.input_data([None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.001)

m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='model_quijote')

m.load(model_path)
seed = random_sequence_from_textfile(path, maxlen)
print(m.generate(600, temperature=1.0, seq_seed=seed))
