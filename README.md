![Python version](https://img.shields.io/badge/python-3.8-blue)

# Lyrics LSTM

LSTM neural network to generate lyrics matching artists' styles and vocabularies that may or may not make sense. 🎵✍️🧠

## Installation

Using `pip`:

```
pip install -r requirements.txt
```

Using `conda`:

```
conda install --file requirements.txt
```

## Usage

### Training

```
python demo_train.py
```

Use `--artist` flag to specify the artist to train a model for. Default is `nicki_minaj`.

Use `--censored` flag to censor explicit lyrics when printing to the terminal. Default is `False`.

Use `--words` flag to specify how many words to generate for the prediction when training. Default is `400`.

```
python demo_train.py --artist taylor_swift --censored --words 100
```

### Prediction

```
python demo_predict.py
```

Use `--artist` flag to specify the artist to get predicted lyrics for. Default is `nicki_minaj`.

Use `--censored` flag to censor explicit lyrics. Default is `False`.

Use `--words` flag to specify how many words to generate. Default is `400`.

```
python demo_predict.py --artist taylor_swift --censored --words 100
```

### Jupyter notebook

Includes cross validation to evaluate model performance and hyperparameter tuning.

```
jupyter notebook
```

## Included files

### Lyrics datasets

Lyrics were taken from [AZLyrics](https://www.azlyrics.com/) and are organised by artists:

- Hayley Kiyoko 👩‍❤️‍💋‍👩
- Nicki Minaj 🐍
- Taylor Swift 👩🏼‍🌾

### Pre-trained models

One pre-trained model and vocabulary dictionary is included per artist using the hyperparameters in the configuration file.

## Possible improvements

- Hyperparameter tuning
- Improved postprocessing (e.g. this does not currently support capitalisation of names)
- Include more artists
