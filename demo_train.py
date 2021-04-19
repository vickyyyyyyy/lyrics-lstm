import sys
import argparse
import torch
from torch.utils.data import DataLoader

from config import Configuration
import preprocess as pp
from models import LyricPredictor
from train import train
from predict import generate_seed_lyrics, predict
from postprocess import postprocess

def main(args):
  '''Example script to train a model on the sample lyrics dataset'''
  c = Configuration()

  print("Hyperparameters: ", c)
  print("Loading data from path: ", c.path)

  lyrics_dataset = pp.read_lyrics_files(c.path)

  tokenized = pp.tokenize(lyrics_dataset)

  x, y, dictionary = pp.preprocess(tokenized, c.window_size)

  training_data = DataLoader(list(zip(x,y)), batch_size=c.train_batch_size, shuffle=True)

  model = LyricPredictor(len(dictionary), c.output_size)

  print("Training model...")

  model, _ , _ = train(model=model, training_data=training_data, num_epochs=c.num_epochs, lr=c.lr, grad_norm=c.grad_max_norm)

  print("Saving model: ", c.model_path)

  torch.save(model, c.model_path)

  print("Generating lyrics...")

  seed_lyrics = generate_seed_lyrics(tokenized, c.window_size, args.censored)

  predicted_lyrics = predict(model, seed_lyrics, dictionary, topk=c.predict_topk)

  predicted_lyrics = postprocess(predicted_lyrics, args.censored)

  print(predicted_lyrics)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--censored", action="store_true")

  main(parser.parse_args(sys.argv[1:]))