import re
import string

from profanity_filter import censor
from preprocess import punctuation_to_transform

def postprocess(lyrics, censored):
  '''
    Format output lyrics
    - Sentence case
    - Remove extra whitespace around punctuation
    - Apply censoring (optional)
  '''
  sentence_case = re.compile(fr'(?<=[.?!\n]\s)(\w+)|(^\w+)|(i[{string.punctuation}| ])')
  lyrics = sentence_case.sub(lambda match: match.group().capitalize(), lyrics)

  lyrics = re.sub(fr" (?=[{punctuation_to_transform}\n])|(?<=\n) ", "", lyrics)

  return censor(lyrics) if censored else lyrics
