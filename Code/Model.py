import torch


class BasicNetwork(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):

        super(BasicNetwork, self).__init__()

        self.input = torch.nn.Linear(input_size, hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.out = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        
        before_hidden = self.input(x)
        after_hidden = self.sigmoid(before_hidden)
        out = self.out(after_hidden)

        return out