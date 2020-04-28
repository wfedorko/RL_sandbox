import torch.nn as nn


class SimpleCNNDueling(nn.Module):
    def __init__(self, config):
        super(SimpleCNNDueling, self).__init__()
        
        self.config=config
        
        self.conv1=nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8 ,stride=4)
        self.conv2=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4 ,stride=2)
        self.conv3=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3 ,stride=1)
        
        self.advantage1 = nn.Linear(7*7*64, config.hidden_layer_size)
        self.advantage2 = nn.Linear(config.hidden_layer_size, config.number_of_outputs)
        
        self.value1 = nn.Linear(7*7*64,config.hidden_layer_size)
        self.value2 = nn.Linear(config.hidden_layer_size,1)
        
        #self.activation=nn.Tanh()
        self.activation=nn.ReLU()
        
    def forward(self, x):
        
        #print('x shape {} and value:'.format(x.shape))
        #print(x.detach().cpu())
        
        if self.config.normalize_image:
            x=x/255.0
        
        output_conv = self.conv1(x)
        output_conv = self.activation(output_conv)
        output_conv = self.conv2(output_conv)
        output_conv = self.activation(output_conv)
        output_conv = self.conv3(output_conv)
        output_conv = self.activation(output_conv)
        
        output_conv = output_conv.view(output_conv.shape[0],-1)
        
        output_advantage=self.advantage1(output_conv)
        output_advantage=self.activation(output_advantage)
        output_advantage=self.advantage2(output_advantage)
        
        output_value=self.value1(output_conv)
        output_value=self.activation(output_value)
        output_value=self.value2(output_value)
        
        #print('output_advantage shape {} and value:'.format(output_advantage.shape))
        #print(output_advantage.detach().cpu())
        
        #print('output_value shape {} and value:'.format(output_value.shape))
        #print(output_value.detach().cpu())
        
        #print('output_advantage.mean shape {} and value:'.format(output_advantage.mean(dim=1,keepdim=True).shape))
        #print(output_advantage.mean(dim=1,keepdim=True).detach().cpu())
        
        #output_final = output_value + output_advantage - output_advantage.mean()
        
        output_final=output_value+output_advantage-output_advantage.mean(dim=1,keepdim=True)
        
        
        return output_final

class SimpleCNNDuelingAllMean(nn.Module):
    def __init__(self, config):
        super(SimpleCNNDuelingAllMean, self).__init__()
        
        
        self.config=config
        
        self.conv1=nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8 ,stride=4)
        self.conv2=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4 ,stride=2)
        self.conv3=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3 ,stride=1)
        
        self.advantage1 = nn.Linear(7*7*64, config.hidden_layer_size)
        self.advantage2 = nn.Linear(config.hidden_layer_size, config.number_of_outputs)
        
        self.value1 = nn.Linear(7*7*64,config.hidden_layer_size)
        self.value2 = nn.Linear(config.hidden_layer_size,1)
        
        #self.activation=nn.Tanh()
        self.activation=nn.ReLU()
        
    def forward(self, x):
        
        #print('in model forward')
        
        #print('x shape {} and value:'.format(x.shape))
        #print(x)
        if self.config.normalize_image:
            x=x/255.0
        
        
        output_conv = self.conv1(x)
        output_conv = self.activation(output_conv)
        output_conv = self.conv2(output_conv)
        output_conv = self.activation(output_conv)
        output_conv = self.conv3(output_conv)
        output_conv = self.activation(output_conv)
        
        #print('conv part done; reshape')
        output_conv = output_conv.view(output_conv.shape[0],-1)
        #print('output_conv shape: {}'.format(output_conv.shape))
        
        #print('reshape done; advantage value')
        output_advantage=self.advantage1(output_conv)
        #print('a1 done')
        output_advantage=self.activation(output_advantage)
        #print('a1relu done')
        output_advantage=self.advantage2(output_advantage)
        #print('a2 done')
        
        output_value=self.value1(output_conv)
        #print('v1 done')
        output_value=self.activation(output_value)
        #print('v1relu done')
        output_value=self.value2(output_value)
        #print('v2 done')
        
        #print('output_advantage shape {} and value:'.format(output_advantage.shape))
        #print(output_advantage.detach().cpu())
        
        #print('output_value shape {} and value:'.format(output_value.shape))
        #print(output_value.detach().cpu())
        
        #print('output_advantage.mean shape {} and value:'.format(output_advantage.mean(dim=1,keepdim=True).shape))
        #print(output_advantage.mean(dim=1,keepdim=True).detach().cpu())
        
        #print('advantage value done; computing final output')
        output_final = output_value + output_advantage - output_advantage.mean()
        
        #output_final=output_value+output_advantage-output_advantage.mean(dim=1,keepdim=True)
        
        #print('returning')
        return output_final
