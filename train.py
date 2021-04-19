import torch.nn as nn
import torch.optim as optim

def train(model, training_data, validation_data=None, num_epochs=10, lr=0.01, grad_norm=5):
  '''
    Train the given model on the training data
    - Use cross entropy loss and Adam optimizer
    - Run the training for num_epochs
    - Use gradient clipping to avoid exploding gradients
    - Calculate training loss
    - Calculate loss on validation data on each epoch (optional)
    - Return trained model and losses
  '''
  loss_function = nn.CrossEntropyLoss()

  optimizer = optim.Adam(model.parameters(), lr=lr)

  model.train()

  training_losses = []
  validation_losses = []

  for epoch in range(num_epochs):
    training_epoch_loss = 0
    validation_epoch_loss = 0

    print("epoch: {} / {}".format(epoch+1, num_epochs))

    for (input_x, label) in training_data:
      prediction = model(input_x)

      loss = loss_function(prediction, label)
      optimizer.zero_grad()
      loss.backward()

      _ = nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_norm)

      optimizer.step()

      training_epoch_loss += loss.detach().item()

    training_epoch_loss = training_epoch_loss / len(training_data)
    training_losses.append(training_epoch_loss)

    if validation_data:
      for (valid_x, valid_label) in validation_data:
        valid_prediction = model(valid_x)
        loss = loss_function(valid_prediction, valid_label)

        validation_epoch_loss += loss.detach().item()

      validation_epoch_loss = validation_epoch_loss / len(validation_data)
      validation_losses.append(validation_epoch_loss)

      print("training loss: {} / validation loss: {}".format(training_epoch_loss, validation_epoch_loss))
    else:
      print("training loss: ", training_epoch_loss)

  return model, training_losses, validation_losses
