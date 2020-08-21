import torch
import torch.nn as nn

class CONFIG:
    '''
    Configuration options for training UNET/UGAN model
    '''
    # Hyperparameters
    in_c = 1  #Input channels
    out_c = 1 #Output channels

    epochs = 30
    batch_size = 5
    criterion = nn.BCELoss() #Loss function
    lr = 0.01 #Learning rate
    device = torch.device("cuda:0") #GPU vs CPU
    to_save = True #Save model and metrics

    saved_model_path = None #If using testing mode
    model_object = None #For testing
config = CONFIG()