from torch import sum, nn

class VAELoss(nn.Module):

    def __init__(self, beta=1, reconstruction_loss=pt.nn.MSELoss()):

        super(VAELoss, self).__init__()
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss

    def forward(self, prediction, ground_truth):

        assert len(prediction) == 3, 'prediction argument must be sequence of reconstruction, mu and sigma'
        reconstruction, mu, sigma = prediction
        reconstruction_loss = self.reconstruction_loss(reconstruction, ground_truth)
        kl_divergence = -0.5 * sum(1 + sigma - mu.pow(2) - sigma.exp())
        loss = reconstruction_loss + self.beta*kl_divergence
        return loss