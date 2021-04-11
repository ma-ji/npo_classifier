#set up environment
import os, torch, pickle, warnings, random, joblib
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm, trange
import tensorflow as tf
warnings.filterwarnings("ignore")
from time import sleep
from joblib import Parallel, delayed
from tlz import partition_all
import itertools
import math

################################### Define functions ##########################
def npoclass(inputs, gpu_core=True, model_path=None, ntee_type='bc', n_jobs=4, backend='multiprocessing'):
    
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    # Check model files.
    if ntee_type=='bc' and model_path==None:
        raise ValueError("Make sure model files/path are correct. Please download from https://jima.me/open/npoclass_model_bc.zip, unzip, and specifiy model_path (default set to None).")
    if ntee_type=='mg' and model_path==None:
        raise ValueError("Make sure model files/path are correct. Please download from https://jima.me/open/npoclass_model_mg.zip, unzip, and specifiy model_path (default set to None).")
        
    # Check ntee type.
    if ntee_type=='bc':
        le_file_name='le_broad_cat.pkl'
    elif ntee_type=='mg':
        le_file_name='le_major_group.pkl'
    else:
        raise ValueError("ntee_type must be 'bc' (broad category) or 'mg' (major group)")

    # Read model and label encoder, if not read.
    global model_loaded, tokenizer_loaded, label_encoder
    try:
        assert model_loaded
        assert tokenizer_loaded
        assert label_encoder
    except:
        #load a pretrained model and tokenizer.
        model_loaded = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer_loaded = BertTokenizer.from_pretrained(model_path)
        # Read label encoder.
        with open(model_path+le_file_name, 'rb') as label_encoder_pkl:
            label_encoder = pickle.load(label_encoder_pkl)
    
    # Select acceleration method.
    if gpu_core==True and torch.cuda.is_available():
        print('There are %d GPU(s) available.' % torch.cuda.device_count(), 'Using GPU:',torch.cuda.get_device_name(0))
        torch.cuda.manual_seed_all(seed_val)
        device = torch.device('cuda')
        model_loaded.cuda()
    else:
        print('No GPU acceleration available or gpu_core=False, using CPU.')
        device = torch.device('cpu')
        model_loaded.cpu()
    print('Encoding inputs ...')
    sleep(.5) # Pause a second for better printing results.
    
    # Encode inputs.
    global func_encode_string, func_encode_string_batch # Define as global, otherwise cannot pickle or very slow.
    def func_encode_string(text_string):
        encoded_dict = tokenizer_loaded.encode_plus(text_string,
                                                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                                    max_length = 256,           # Pad & truncate all sentences.
                                                    truncation=True,
                                                    pad_to_max_length = True,
                                                    return_attention_mask = True,   # Construct attn. masks.
                                                    return_tensors = 'pt',     # Return pytorch tensors.
                                                   )
        return encoded_dict
    def func_encode_string_batch(text_strings):
        encoded_dicts=[]
        for text_string in text_strings:
            encoded_dicts+=[func_encode_string(text_string)]
        return encoded_dicts

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    # Encode input string(s).
    if type(inputs)==list:
        if backend=='multiprocessing':
            encoded_outputs=Parallel(n_jobs=n_jobs, backend="multiprocessing", pre_dispatch=n_jobs, verbose=1)(delayed(func_encode_string)(text_string) for text_string in inputs)
            for encoded_output in encoded_outputs:
                # Add the encoded sentence to the list.
                input_ids.append(encoded_output['input_ids'])
                # And its attention mask (simply differentiates padding from non-padding).
                attention_masks.append(encoded_output['attention_mask'])
        elif backend=='sequential':
            for text_string in tqdm(inputs):
                encoded_output=func_encode_string(text_string)
                # Add the encoded sentence to the list.
                input_ids.append(encoded_output['input_ids'])
                # And its attention mask (simply differentiates padding from non-padding).
                attention_masks.append(encoded_output['attention_mask'])
        elif backend=='dask':
            with joblib.parallel_backend('dask'):
                n_jobs=len(client.scheduler_info()['workers']) # Get # works.
                string_chunks = partition_all(math.ceil(len(inputs)/n_jobs), inputs)  # Collect into groups of size 1000
                encoded_outputs=Parallel(n_jobs=-1, batch_size='auto', verbose=1)(delayed(func_encode_string_batch)(text_strings) for text_strings in string_chunks)
                encoded_outputs=itertools.chain(*encoded_outputs)
            for encoded_output in encoded_outputs:
                # Add the encoded sentence to the list.
                input_ids.append(encoded_output['input_ids'])
                # And its attention mask (simply differentiates padding from non-padding).
                attention_masks.append(encoded_output['attention_mask'])           
    if type(inputs)==str:
        encoded_output=func_encode_string(inputs)
        input_ids=[encoded_output['input_ids']]
        attention_masks=[encoded_output['attention_mask']]

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Prepare dataloader for efficient calculation.
    batch_size = 64
    pred_data = TensorDataset(input_ids, attention_masks)
    pred_sampler = SequentialSampler(pred_data)
    pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=batch_size)

    # Start prediction.
    model_loaded.eval()
    logits_all=[]
    print('Predicting categories ...')
    sleep(.5) # Pause a second for better printing results.
    for batch in tqdm(pred_dataloader):
        # Add batch to the pre-chosen device
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            outputs = model_loaded(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits_all+=outputs[0].tolist()

    # Calculate probabilities of logitcs.
    logits_prob=tf.nn.sigmoid(logits_all).numpy().tolist()
    # Find the positions of max values in logits.
    logits_max=np.argmax(logits_prob, axis=1)
    # Transfer to labels.
    logits_labels=label_encoder.inverse_transform(logits_max)
    
    # Compile results to be returned.
    result_list=[]
    for list_index in range(0, len(logits_labels)):
        result_dict={}
        result_dict['recommended']=logits_labels[list_index]
        conf_prob=logits_prob[list_index][logits_max[list_index]]
        if conf_prob>=.99:
            result_dict['confidence']='high (>=.99)'
        elif conf_prob>=.95:
            result_dict['confidence']='medium (<.99|>=.95)'
        else:
            result_dict['confidence']='low (<.95)'
        prob_dict={}
        for label_index in range(0, len(label_encoder.classes_)):
            prob_dict[label_encoder.classes_[label_index]]=logits_prob[list_index][label_index]
        result_dict['probabilities']=prob_dict
        result_list+=[result_dict]

    return result_list