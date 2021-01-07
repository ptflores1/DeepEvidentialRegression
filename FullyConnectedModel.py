import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNormalGamma(nn.Module):
    def __init__(self, n_input, n_out_tasks=1):
        super(FCNormalGamma, self).__init__()
        self.n_in = n_input
        self.n_out = 4 * n_out_tasks
        self.n_tasks = n_out_tasks
        self.l1 = nn.Linear(self.n_in, self.n_out)

    def forward(self, x):
        x = self.l1(x)
        if len(x.shape) == 1:
            gamma, lognu, logalpha, logbeta = torch.split(
                x, self.n_tasks, dim=0)
        else:
            gamma, lognu, logalpha, logbeta = torch.split(
                x, self.n_tasks, dim=1)

        nu = F.softplus(lognu)
        alpha = F.softplus(logalpha) + 1.
        beta = F.softplus(logbeta)

        return torch.stack([gamma, nu, alpha, beta]).to(x.device)


class EvidentialRegression(nn.Module):
    def __init__(self, input_size, num_neurons=50, num_layers=1, activation=F.relu):
        super(EvidentialRegression, self).__init__()
        self.activation = activation
        self.num_layers = num_layers
        self.in_fc = nn.Linear(input_size, num_neurons)
        self.fcs = nn.ModuleList(
            [nn.Linear(num_neurons, num_neurons) for _ in range(num_layers-1)])
        self.fcNormalGamma = FCNormalGamma(num_neurons)

    def forward(self, x):
        x = self.activation(self.in_fc(x))
        for hidden_layer in self.fcs:
            x = self.activation(hidden_layer(x))
        x = self.fcNormalGamma(x)
        return x.squeeze()

if __name__ == "__main__":
    model = EvidentialRegression(1)
    x = torch.rand(64, 1)
    print(model(x), model(x).size())
