import sys
import torch

from parser import get_argument_parser
from config import Configuration
import preprocess as pp
from predict import generate_seed_lyrics, predict
from postprocess import postprocess

def main(args):
  '''Example script to run prediction on a pre-trained model with sample lyrics dataset'''
  c = Configuration()

  if args.artist:
    c.set_artist(args.artist)

  print("Artist:", c.artist.replace("_", " ").title())

  lyrics_dataset = pp.read_lyrics_files(c.path)

  dictionary = torch.load(open(c.dictionary_path, 'rb'))

  print("Vocabulary size: ", len(dictionary))
  print("----------------------------")

  tokenized = pp.tokenize(lyrics_dataset)

  seed_lyrics = generate_seed_lyrics(tokenized, c.window_size, args.censored)

  model = torch.load(open(c.model_path, 'rb'))

  predicted_lyrics = predict(model, seed_lyrics, dictionary, num_words=args.words, topk=c.predict_topk)

  predicted_lyrics = postprocess(predicted_lyrics, args.censored)

  print(predicted_lyrics)

if __name__ == "__main__":
  parser = get_argument_parser()

  main(parser.parse_args(sys.argv[1:]))
