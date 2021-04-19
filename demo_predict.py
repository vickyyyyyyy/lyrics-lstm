import sys
import argparse
import torch

from config import Configuration
import preprocess as pp
from predict import generate_seed_lyrics, predict
from postprocess import postprocess

def main(args):
  '''Example script to run prediction on a pre-trained model with sample lyrics dataset'''
  c = Configuration()

  lyrics_dataset = pp.read_lyrics_files(c.path)

  tokenized = pp.tokenize(lyrics_dataset)

  seed_lyrics = generate_seed_lyrics(tokenized, c.window_size, args.censored)

  model = torch.load(open(c.model_path, 'rb'))

  dictionary = torch.load(open(c.dictionary_path, 'rb'))

  predicted_lyrics = predict(model, seed_lyrics, dictionary, num_words=args.num_words, topk=c.predict_topk)

  predicted_lyrics = postprocess(predicted_lyrics, args.censored)

  print(predicted_lyrics)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--censored", action="store_true")
  parser.add_argument("--num_words", type=int, default=400)

  main(parser.parse_args(sys.argv[1:]))
