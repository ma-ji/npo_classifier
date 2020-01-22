################################### Define reproducibility ##########################
# # Force using CPU.
# # https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# obtain reproducible results

import numpy as np
import tensorflow as tf
import random as rn

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


################################### Import dependencies ##########################
from spellchecker import SpellChecker
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.models import load_model
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_list=[str.upper(s) for s in stopwords.words('english')+list(string.punctuation)]
import warnings
warnings.simplefilter('ignore')
import numpy as np
from keras.utils import np_utils
import os
import pandas as pd
import tensorflow as tf
from multiprocessing import Pool
p=Pool()


################################### Load saved models and classes ##########################
model_broad_cat=load_model('https://raw.githubusercontent.com/"[anonymized]"/npo_classifier/master/output/broad_category_model.h5')
model_major_group=load_model('https://raw.githubusercontent.com/"[anonymized]"/npo_classifier/master/output/major_group_model.h5')
with open('https://raw.githubusercontent.com/"[anonymized]"/npo_classifier/master/output/tokenizer.pkl', 'rb') as tokenizer_pkl:
    tokenizer = pickle.load(tokenizer_pkl)
with open('https://raw.githubusercontent.com/"[anonymized]"/npo_classifier/master/output/lb_broad_cat.pkl', 'rb') as lb_broad_cat_pkl:
    lb_broad_cat = pickle.load(lb_broad_cat_pkl)
with open('https://raw.githubusercontent.com/"[anonymized]"/npo_classifier/master/output/lb_major_group.pkl', 'rb') as lb_major_group_pkl:
    lb_major_group = pickle.load(lb_major_group_pkl)

# String/String list input --> a list of string token list(s) --> spellchecking (parallel) --> predict class (serial).


################################### Define functions ##########################
def npoclass(string_input=None):
    ## Define local function.
    # Spell check function. Return corrected word if unknown; return original word if known.
    def spellcheck(input_string):
        if type(input_string)==str:
            word_token_list=nltk.word_tokenize(input_string)
            return [s.upper() for s in p.map(SpellChecker().correction, word_token_list)]
        elif type(input_string)==list:
            word_token_list_list=[nltk.word_tokenize(string) for string in input_string]
            word_token_list_list_chk=[]
            for word_token_list in word_token_list_list:
                word_token_list_list_chk+=[[s.upper() for s in p.map(SpellChecker().correction, word_token_list)]]
            return word_token_list_list_chk
        else:
            raise NameError('Input must be a string or a list of strings.')
    
    result_dict={}
    # Text to sequences.
    # seq_encoding_text_test=tokenizer.texts_to_sequences(spellcheck(input_string_list))
    seq_encoding_text=tokenizer.texts_to_sequences(spellcheck(string_input))
    # Pads sequences to the same length (i.e., prepare matrix).
    x_text=pad_sequences(sequences=seq_encoding_text,
                        maxlen=46612, # Max length of the sequence.
                        dtype = "int32", padding = "post", truncating = "post", 
                        value = 0 # Zero is used for representing None or Unknown.
                         )
    # Predict.
    y_prob=model_major_group.predict(x_text)
    result_dict['major_group_label']=lb_major_group.inverse_transform(np_utils.to_categorical(y_prob.argmax(axis=-1))).tolist()
    result_dict['major_group_prob']=[s.max() for s in y_prob]
    y_prob=model_broad_cat.predict(x_text)
    result_dict['broad_category_label']=lb_broad_cat.inverse_transform(np_utils.to_categorical(y_prob.argmax(axis=-1))).tolist()
    result_dict['broad_category_prob']=[s.max() for s in y_prob]
    return result_dict
