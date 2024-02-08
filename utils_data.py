
# Utility functions for data manipulation and preparation 

import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from torchvision import transforms, utils

# TODO: Complete the dataset definition
class GPT2Dataset(Dataset):
  def __init__(self, df):
    self.encodings = df['encodings'].to_list()
    self.sum_idx = df['text_len'].to_list()
  
  def __len__(self):
    return len(self.sum_idx)

  def __getitem__(self, idx):
    text = torch.tensor() # exract input_ids from encodings
    attn_mask = torch.tensor() # exract mask from encodings
    s_idx = self.sum_idx[idx] + 2 # add bos and cls
    out = {'text': text, 'mask': attn_mask, 's_idx': s_idx}
    return out 

def get_gpt2_dataset(train, val):

  train_dataset = GPT2Dataset(train)
  val_dataset = GPT2Dataset(val)

  return train_dataset, val_dataset


def short_text(text, len):
	text = text.split()
	#print(len(text))
	#len= len-1
	s_text = text[0:len]
	s_text = ' '.join(s_text)
	return  s_text


def process_dataframe(df, max_text, max_sum):

	df['text'] = df['text'].apply(lambda x: short_text(x, max_text))
	df['summary'] = df['summary'].apply(lambda x: short_text(x, max_sum))
	df['text_len'] = df['text'].apply(lambda x: len(x.split()))
	#print(df['summary'].str.split().str.len())
	#df['summary'] = df['summary'].apply(lambda x: ' <CLS> ' + x) # do this step in data loader
	#df['ts'] = df[['text', 'summary']].apply(lambda x: ''.join(x), axis=1)
	#print(df['ts'].str.split().str.len())

	return df

def split_data (df, sr):
	train, val_test = train_test_split(df,test_size=sr)
	val, test = train_test_split(val_test,test_size=0.5)
	return train, val, test

def process_data(file, max_text, max_sum, sr):
	
	# load into a data frame
	df = pd.read_csv(file)  
	df = process_dataframe(df, max_text, max_sum)
	train, val, test = split_data(df, sr)
	print('train size: {}'.format(len(train)))
	print('val size: {}'.format(len(val)))
	print('test size: {}'.format(len(test)))
	print('test head:\n{}'.format(test.head(1)))

	return train, val, test
	