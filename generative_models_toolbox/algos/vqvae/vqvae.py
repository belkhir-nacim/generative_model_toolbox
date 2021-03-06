import torch
import torch.nn as nn
import torch.nn.functional as F
from .functions import vq, vq_st, vq_st_i, vq_st_ori, vq_st_i_ori
from generative_models_toolbox.utils import weights_init


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        batch_size, height, width, channels = input_shape
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) + torch.sum(self._embedding.weight ** 2,
                                                                                 dim=1) - 2 * torch.matmul(flat_input,
                                                                                                           self._embedding.weight.t()))
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        encoding_indices = encoding_indices.view(batch_size, height, width)
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        batch_size, height, width, channels = input_shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True) + torch.sum(self._embedding.weight ** 2,
                                                                                 dim=1) - 2 * torch.matmul(flat_input,
                                                                                                           self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        encoding_indices = encoding_indices.view(batch_size, height, width)
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(nn.ReLU(True),
                                    nn.Conv2d(in_channels=in_channels, out_channels=num_residual_hiddens, kernel_size=3,
                                              stride=1, padding=1),
                                    nn.InstanceNorm2d(num_residual_hiddens), nn.ReLU(True),
                                    nn.Conv2d(in_channels=num_residual_hiddens, out_channels=num_hiddens, kernel_size=1,
                                              stride=1),
                                    nn.InstanceNorm2d(num_residual_hiddens))

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [Residual(in_channels, num_hiddens, num_residual_hiddens) for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, nb_downsample_blocks=2, ):
        super().__init__()
        layers = []
        for i in range(nb_downsample_blocks):
            layers.append(
                nn.Conv2d(in_channels=in_channels if i == 0 else num_hiddens, out_channels=num_hiddens, kernel_size=4,
                          stride=2, padding=1, ))
            layers.append(nn.InstanceNorm2d(num_hiddens))
            layers.append(nn.ReLU(True))
        self.downsample = nn.Sequential(*layers)
        self._conv = nn.Conv2d(in_channels=num_hiddens, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self.downsample(inputs)
        x = self._conv(x)
        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, nb_upsample_blocks=2,
                 out_channels=3):
        super(Decoder, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=num_hiddens, kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens, num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        layers = []
        for i in range(nb_upsample_blocks):
            last = i == nb_upsample_blocks - 1
            out = out_channels if last else num_hiddens
            layers.append(
                nn.ConvTranspose2d(in_channels=num_hiddens, out_channels=out, kernel_size=4, stride=2, padding=1))
            if not last:
                layers.append(nn.InstanceNorm2d(num_hiddens))
                layers.append(nn.ReLU(True))
        self.upsample = nn.Sequential(*layers)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        return self.upsample(x)


class VQVAEModel(nn.Module):
    def __init__(self, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, num_embeddings=51,
                 embedding_dim=64, commitment_cost=0.25, decay=0.99, nb_channels=1, nb_blocks=2):
        super().__init__()
        self._encoder = Encoder(nb_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                                nb_downsample_blocks=nb_blocks)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self._decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens,
                                out_channels=nb_channels, nb_upsample_blocks=nb_blocks)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity

    def encode(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, encoding_indices = self._vq_vae(z)
        return encoding_indices

    def reconstruct_from_code(self, encoding_indices):
        batch_size, height, width = encoding_indices.shape
        encoding_indices = encoding_indices.view(-1)
        encodings = torch.nn.functional.one_hot( encoding_indices, num_classes=self._vq_vae._num_embeddings)
        encodings = encodings.float()
        quantized = torch.matmul(encodings, self._vq_vae._embedding.weight)
        quantized = quantized.view(batch_size, height, width, self._vq_vae._embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        x_recon = self._decoder(quantized)
        return x_recon

    @property
    def num_embeddings(self):
        return self._vq_vae._num_embeddings
