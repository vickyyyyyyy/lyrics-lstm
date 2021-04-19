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

Use `--censored` flag to censor explicit lyrics when printing to the terminal.

```
python demo_train.py --censored
```

### Prediction

```
python demo_predict.py
```

Use `--censored` flag to censor explicit lyrics.

Use `--num_words` flag to specify how many words to generate.

```
python demo_predict.py --censored --num_words 100
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

One pre-trained model is included per artist using the hyperparameters in the configuration file.

## Possible improvements

- Hyperparameter tuning
- Improved postprocessing (e.g. this does not currently support capitalisation of names)
- Include more artists