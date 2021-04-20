import argparse

def get_argument_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--artist", type=str)
  parser.add_argument("--censored", action="store_true")
  parser.add_argument("--words", type=int, default=400)

  return parser