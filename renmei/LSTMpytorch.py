#---------------------------任梅---------------------
#5.26利用pytorch构建了一个LSTM 模型
import torch
input_size = 784
hidden_size = 64
output_size = 784  # dimenskion 784 = (28*28) --> 64 --> 784 = (28*28)
epochs = 5
batch_size = 128
num_layers=10
class LSTM(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(LSTM,self),__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=torch.nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_size=true)
    def forward(self,input):
        self.hidden = self.initHidden(input.size(0))
#input应该为(batch_size,output_size,input_szie)
        out, self.hidden = lstm(input, self.hidden)
        return out, self.hidden

    def initHidden(self, batch_size):
        if self.lstm.bidirectional:
            return (torch.rand(self.num_layers * 2, batch_size, self.hidden_size),
                    torch.rand(self.num_layers * 2, batch_size, self.hidden_size))
        else:
            return (torch.rand(self.num_layers, batch_size, self.hidden_size),
                    torch.rand(self.num_layers, batch_size, self.hidden_size))
model =LSTM(input_size,hidden_size,num_layers)

