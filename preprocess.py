import os
import string
import re
import numpy as np
import torch

punctuation_to_transform = ",.?!"

def read_lyrics_files(path):
  '''Return content of all lyrics files in path as a string'''
  lyrics_dataset = ""

  songs = os.listdir(path)

  for lyric_file in songs:
    with open(os.path.join(path, lyric_file), encoding="utf-8", mode="r") as file:
      data = file.read()
      lyrics_dataset += data + "\n"

  print("Number of songs: {}".format(len(songs)))

  return lyrics_dataset

def tokenize(lyrics_dataset):
  '''
    Format raw lyrics string
    - Lowercase
    - Left pad punctuation_to_transform with space
    - Remove all other punctuation except for punctuation_to_keep
    - Replace newlines with a token
    - Split string into list of words (including newline as a word)
  '''
  punctuation_to_keep = "'-"
  new_line_token = " $newline$ "

  lyrics = lyrics_dataset.lower()

  lyrics = re.sub(fr"(?=[{punctuation_to_transform}])", " ", lyrics)

  punctuation_to_remove = re.sub(fr"[{punctuation_to_keep}]|[{punctuation_to_transform}]", "", string.punctuation)

  lyrics = re.sub(fr"[{punctuation_to_remove}]", "", lyrics)

  lyrics = lyrics.replace("\n", new_line_token)

  tokenized = lyrics.split()

  tokenized = [t.replace(new_line_token.strip(), "\n") for t in tokenized]

  return tokenized

def get_dictionary(tokenized):
  '''Return unique words from tokenized'''
  return list(dict.fromkeys(tokenized))

def preprocess(tokenized, window_size):
  '''
    Transform tokenized dataset into Tensor format
    - Convert list of word strings to list of dictionary indices
    - Slide window along the list to create sequences of length window_size
    - Split each sequence into input X and output y where y is the next word given input X
    - Return all inputs, all outputs and the dictionary
  '''
  sequences = []
  dictionary = get_dictionary(tokenized)

  transformed = [dictionary.index(token) for token in tokenized]

  for i in range(len(transformed) - window_size):
    sequences.append(transformed[i:i+window_size])

  X = [sequence[:-1] for sequence in sequences]
  y = [sequence[-1] for sequence in sequences]

  print("Vocabulary size: ", len(dictionary))

  X = torch.Tensor(X).long()
  y = torch.Tensor(y).long()

  return X, y, dictionary

def one_hot(label, n_classes):
  '''Convert label to one hot encoding, given n_classes'''
  targets = np.array(label)

  return np.eye(n_classes)[targets]
