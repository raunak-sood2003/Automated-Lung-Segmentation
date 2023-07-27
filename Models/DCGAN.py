class Generator(nn.Module):
    def __init__(self, in_channels, batch_size, kernel_size = (3, 3), pool_size = (2, 2), padding = 0, conv_stride = 1, pool_stride = 2):
        super(Generator, self).__init__()
        
        self.convblock1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, 16, kernel_size, conv_stride, padding)),
            ('activation1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(pool_size, pool_stride))
        ]))
        
        self.convblock2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(16, 32, kernel_size, conv_stride, padding)),
            ('activation2', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(pool_size, pool_stride))
        ]))
        
        self.convblock3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(32, 64, kernel_size, conv_stride, padding)),
            ('activation3', nn.ReLU()),
            ('maxpool3', nn.MaxPool2d(pool_size, pool_stride))
        ]))
        
        self.upblock = nn.Sequential(OrderedDict([
            ('upsample1', nn.Upsample((124, 124))),
            ('transposed_conv1', nn.ConvTranspose2d(64, 32, kernel_size, conv_stride, padding)),
            ('activation1', nn.ReLU()),
            ('upsample2', nn.Upsample((253, 253))),
            ('transposed_conv2', nn.ConvTranspose2d(32, 16, kernel_size, conv_stride, padding)),
            ('activation2', nn.ReLU()),
            ('upsample3', nn.Upsample((510, 510))),
            ('transposed_conv3', nn.ConvTranspose2d(16, 1, kernel_size, conv_stride, padding)),
            ('activation2', nn.Sigmoid())
        ]))
        
    def forward(self, X):
        X = self.convblock1(X)
        X = self.convblock2(X)
        X = self.convblock3(X)
        X = self.upblock(X)
        return X
      

class Discriminator(nn.Module):
  def __init__(self, in_channels, batch_size, kernel_size = (3, 3), pool_size = (2, 2), padding = 0, conv_stride = 1, pool_stride = 2):
      super(Discriminator, self).__init__()
      
      self.convblock1 = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(in_channels, 16, kernel_size, conv_stride, padding)),
          ('activation1', nn.ReLU()),
          ('maxpool1', nn.MaxPool2d(pool_size, pool_stride))
      ]))
      
      self.convblock2 = nn.Sequential(OrderedDict([
          ('conv2', nn.Conv2d(16, 32, kernel_size, conv_stride, padding)),
          ('activation2', nn.ReLU()),
          ('maxpool2', nn.MaxPool2d(pool_size, pool_stride))
      ]))
      
      self.convblock3 = nn.Sequential(OrderedDict([
          ('conv3', nn.Conv2d(32, 64, kernel_size, conv_stride, padding)),
          ('activation3', nn.ReLU()),
          ('maxpool3', nn.MaxPool2d(pool_size, pool_stride))
      ]))
      
      self.linear = nn.Sequential(OrderedDict([
          ('linear1', nn.Linear(64*62*62, 256)),
          ('activation1', nn.ReLU()),
          ('linear2', nn.Linear(256, 64)),
          ('activation2', nn.ReLU()),
          ('linear3', nn.Linear(64, 1)),
          ('activation3', nn.Sigmoid())
          
      ]))
      
  def forward(self, X):
      
      X = self.convblock1(X)
      X = self.convblock2(X)
      X = self.convblock3(X)
      X = X.view(-1, 64*62*62)
      X = self.linear(X)
      return X
