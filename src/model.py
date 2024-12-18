import torch
import torch.nn as nn

class ConvMLP(nn.Module):
    def __init__(self, params):
        '''
        Args:
            params: dictionary with the following
                input_channels: number of input channels for convolutional layer (int)
                conv_out_channels: number of output channels for convolutional layer (int)
                conv_kernel_size: kernel size for convolutional layer (int or tuple)
                pool_kernel_size: kernel size for pooling layer (int or tuple)
                l_i_size: input size for first fully connected layer after flattening (int)
                h_lx_size: hidden layer x size (int)
                l_o_size: output layer size (int)
        '''
        super(ConvMLP, self).__init__()
        
        # Convolutional Layers with Batch Normalization and LeakyReLU
        self.conv1 = nn.Conv2d(in_channels=params['input_channels'], 
                               out_channels=params['conv1_out_channels'], 
                               kernel_size=params['conv_kernel_size'])
        self.bn1 = nn.BatchNorm2d(params['conv1_out_channels'])
        
        self.conv2 = nn.Conv2d(in_channels=params['conv1_out_channels'], 
                               out_channels=params['conv2_out_channels'], 
                               kernel_size=params['conv_kernel_size'])
        self.bn2 = nn.BatchNorm2d(params['conv2_out_channels'])
        
        self.pool = nn.MaxPool2d(kernel_size=params['pool_kernel_size'])
        
        # Fully Connected Layers (MLP Part)
        self.fc1 = nn.Linear(params['l_i_size'], params['h_l1_size'])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(params['h_l1_size'], params['h_l2_size'])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(params['h_l2_size'], params['h_l3_size'])
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(params['h_l3_size'], params['h_l4_size'])
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(params['h_l4_size'], params['l_o_size'])
        self.softmax = nn.Softmax(dim=1)
                
        # Initialize weights
        self.apply(initialization)

    def forward(self, x):
        '''
        Args:
            x: input tensor (e.g., a batch of images)
        '''
        # Pass through convolutional layers with LeakyReLU and BatchNorm
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.1)(x)  # LeakyReLU with negative slope 0.1
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.1)(x)
        x = self.pool(x)
        
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, start_dim=1)
        
        # Pass through fully connected layers
        x = self.fc1(x)
        x = nn.LeakyReLU(0.1)(x)
        x = self.fc2(x)
        x = nn.LeakyReLU(0.1)(x)
        x = self.fc3(x)
        x = nn.LeakyReLU(0.1)(x)
        x = self.fc4(x)
        x = nn.LeakyReLU(0.1)(x)
        x = self.fc5(x)
        
        return x

def initialization(model):
    '''
    Args:
        model: model to be initialized
    '''
    if isinstance(model, nn.Linear):
        nn.init.kaiming_normal_(model.weight)
        nn.init.zeros_(model.bias)
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(model.bias)