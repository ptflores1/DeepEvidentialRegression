import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def NIG_NLL(y, gamma, nu, alpha, beta):
    omega = 2 * beta * (1 + nu)
    nll = 0.5 * (np.pi / nu).log()\
        - alpha*omega.log()\
        + (alpha + 0.5) * (nu * (y - gamma) ** 2 + omega).log() \
        + torch.lgamma(alpha) \
        - torch.lgamma(alpha + 0.5)
    return nll.mean()


def NIG_Regularization(y, gamma, nu, alpha):
    error = (y - gamma).abs()
    evidence = 2 * nu + alpha
    return (error * evidence).mean()


def EvidentialRegressionLoss(y, evidential_output):
    gamma, nu, alpha, beta = evidential_output
    loss_nll = NIG_NLL(y, gamma, nu, alpha, beta)
    loss_reg = NIG_Regularization(y, gamma, nu, alpha)
    return loss_nll, loss_reg


def RMSELoss(y, preds):
    rmse = torch.sqrt(((y-preds)**2).mean())
    return rmse


class EvidentialTrainer:

    @staticmethod
    def get_data_loader(x, y, batch_size):
        data = torch.cat([x, y], dim=1)
        data_loader = InfiniteDataLoader(
            data, batch_size=batch_size, shuffle=True)
        return data_loader

    @staticmethod
    def get_batch(x, y, batch_size):
         idx = np.random.choice(x.shape[0], batch_size, replace=False)
         x_ = x[idx]
         y_ = y[idx]
         return x_, y_

    def __init__(self, model, learning_rate=1e-3, lambda_coef=0.0, epsilon=1e-2, maxi_rate=1e-4, storage_options=None):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.learning_rate = learning_rate
        self.maxi_rate = maxi_rate
        self.epsilon = epsilon
        self.storage_options = storage_options

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, eps=self.epsilon)
        self.lambda_coef = torch.tensor([lambda_coef]).to(device=self.device)

        self.min_rmse=float('inf')
        self.min_nll=float('inf')
        self.min_tloss=float('inf')

        
    def loss_function(self, y, evidential_output):
        gamma, nu, alpha, beta = evidential_output
        nll_loss = NIG_NLL(y, gamma, nu, alpha, beta)
        reg_loss = NIG_Regularization(y, gamma, nu, alpha)
        loss = nll_loss + self.lambda_coef * (reg_loss - self.epsilon)
        return loss, nll_loss, reg_loss

    def train_step(self, x, y):
        evidential_output = self.model(x)
        loss, nll_loss, reg_loss = self.loss_function(y, evidential_output)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.lambda_coef += self.maxi_rate * (reg_loss - self.epsilon)

    @torch.no_grad()
    def evaluate(self, x, y):
        self.model.eval()

        evidential_output = self.model(x)
        rmse = RMSELoss(y, evidential_output[0])
        loss, nll_loss, reg_loss = self.loss_function(y, evidential_output)

        self.model.train()

        return loss, nll_loss, reg_loss, rmse
    
    def save(self):
        path = f"./{self.storage_options['folder']}/{self.storage_options['name']}"
        torch.save(self.model.state_dict(), path)

    def train(self, x_train, y_train, x_test, y_test, y_test_scale, batch_size=128, iters=5000, verbose=True):
        for it in range(iters):
            x_batch, y_batch = EvidentialTrainer.get_batch(x_train, y_train, batch_size)
            x_batch = x_batch.to(device=self.device)
            y_batch = y_batch.squeeze(-1).to(device=self.device)
            self.train_step(x_batch, y_batch)

            if it % 100 == 0:
                x_test_batch, y_test_batch = EvidentialTrainer.get_batch(x_test, y_test, min(100, x_test.size(0)))
                x_test_batch = x_test_batch.to(device=self.device)
                y_test_batch = y_test_batch.squeeze(-1).to(device=self.device)

                tot_loss, nll, reg_loss, rmse = self.evaluate(
                    x_test_batch, y_test_batch)
                nll += np.log(y_test_scale)
                rmse *= y_test_scale

                if tot_loss < self.min_tloss:
                    self.save()

                self.min_rmse = min([self.min_rmse, rmse])
                self.min_nll = min([self.min_nll, nll])
                self.min_tloss = min([self.min_tloss, tot_loss])
                if verbose:
                    print("[{}]  RMSE: {:.4f} \t NLL: {:.4f} \t loss: {:.4f} \t reg_loss: {:.4f} \t lambda: {:.2f}".format(
                        it, self.min_rmse, self.min_nll, self.min_tloss.item(), reg_loss, self.lambda_coef.item()))

        return self.min_rmse, self.min_nll


if __name__ == "__main__":
    import data_loader
    from FullyConnectedModel import EvidentialRegression
    (x_train_boston, y_train_boston), (x_test_boston,
                                       y_test_boston), y_scale_boston = data_loader.load_dataset("boston", return_as_tensor=True)
    model = EvidentialRegression(x_train_boston.size(1))
    trainer = EvidentialTrainer(model)
    trainer.train(x_train_boston, y_train_boston, x_test_boston, y_test_boston, y_scale_boston.item())
