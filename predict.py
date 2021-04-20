import random
import torch
import numpy as np

from profanity_filter import censor

def generate_seed_lyrics(tokenized, window_size, censored=False):
  '''Return random sequence of length window_size from tokenized version of lyrics dataset'''
  seed_index = random.randint(0, len(tokenized)-window_size)
  lyrics = tokenized[seed_index:seed_index+window_size]

  print_lyrics = censor(" ".join(lyrics)) if censored else " ".join(lyrics)

  print("Seed lyrics: ", repr(print_lyrics))
  print("----------------------------")

  return lyrics

def predict(model, lyrics, dictionary, num_words=100, topk=5):
  '''
    Repeatedly run a lyric sequence through the given model to generate a sequence of length num_words
    Each next word is randomly selected from the top k predictions for better variety
  '''
  valid_x = torch.Tensor([[dictionary.index(initial_word) for initial_word in lyrics]]).long()

  for _ in range(num_words):
    prediction = model(valid_x)

    _, top_choices = torch.topk(prediction, k=topk)

    choice_index = np.random.choice(top_choices[0])

    next_word = dictionary[choice_index]

    lyrics.append(next_word)

    valid_x = torch.cat((valid_x[:,1:], torch.Tensor(choice_index.reshape(1,1)).long()), 1)

  if num_words < len(lyrics):
    lyrics = lyrics[0:num_words]

  return " ".join(lyrics)
