import torch


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


myModule = MyModule()
x = torch.tensor(1.0)
output = myModule(x)
print(output)