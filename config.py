class Configuration():
  '''Configuration to store constants and hyperparameters'''
  def __init__(self):
    self.artist = "nicki_minaj"
    self.path = "lyrics/" + self.artist
    self.model_path = "ckpt/model_" + self.artist
    self.dictionary_path = "ckpt/dictionary_"  + self.artist
    self.window_size = 4
    self.train_batch_size = 64
    self.valid_batch_size = 64
    self.output_size = 256
    self.num_epochs = 15
    self.lr = 0.001
    self.grad_max_norm = 5
    self.predict_topk = 5

  def __str__(self):
    '''Return configuration as a string'''
    return str(self.__dict__)
