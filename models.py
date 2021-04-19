import torch.nn as nn

class LyricPredictor(nn.Module):
  '''LSTM neural network that predicts the next word given a sequence of words'''
  def __init__(self, dictionary_size, output_size):
    super().__init__()
    self.embedding = nn.Embedding(num_embeddings=dictionary_size, embedding_dim=output_size)
  
    self.lstm = nn.LSTM(input_size=output_size, hidden_size=output_size)

    self.linear = nn.Linear(in_features=output_size, out_features=dictionary_size)

  def forward(self, x):
    '''Run input x through the model layers'''
    x = self.embedding(x)

    x, _ = self.lstm(x)

    x = x[:, -1, :]

    x = self.linear(x)

    return x
