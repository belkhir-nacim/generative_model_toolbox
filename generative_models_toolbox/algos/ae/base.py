import numpy as np
import torch.nn as nn
from collections import Sequence
from functools import partial



class ConvAE(nn.Module):
    """
    Class representing a convolutional autoencoder. The architecture of the AE is highly customizable with
    regards to the number of layers, number of residual blocks, the functions used for the various types on convolution
    and the number of channels at each step in the autoencoder.
    In this sense it is an attempt at a semi-general constructor for an AE.
    """

    def __init__(self, n_layers, n_residual, channel_factor, max_channels, input_channels, channels,
                 down_conv, up_conv, res_block, final_norm=True):
        """
        the architecture of the AE is dynamically build based on the arguments passed to this init method. It is
        important to note that there are two mechanisms for defining the number of layers and
        channels in each layer:
        - First method:
            The more traditional of the two methods. Specifying channel_factor, n_layers and max_channels will result
            in a network that hold n_layers downsampling layers and doubling the amount of channels with each, starting
            from channel_factor channels after an initial convolution. If the number of channels after another round of
            doubling would exceed max_channels it is set to max_channels instad. In the decoder the number of channels
            will be halved with every upsampling layer.
        - Second method:
            Passing a dict  of the form {'encoder': [list of int], 'decoder': [list of int]} to channels will result in
            a network which has len(channels['encoder']) downsampling layers in the encoder, where each layer has the
            number of channels specified in the list. The same is true for the decoder but with upsampling layers in
            that case. If only one of the keys ('encoder' or 'decoder') is present in the dict the missing one will be
            constructed according to the first method.

        :param n_layers: number of down- and upsampling layers in the encoder and decoder respectively
        :param n_residual: tuple that controls the residual blocks in the network. The first entry specifies the amount
        of residual blocks in between down- or upsampling layers. The second specifies the amount of convolutions in
        each block. Alternatively, two tuples can be passed. The first will be used for the encoder and the second for
        the decoder.
        :param channel_factor: number of channels of the tensor after the inital convolution.
        :param max_channels: maximum amount of channels that should be present in the network
        :param input_channels: number of channels in the input tensor
        :param channels: dict with keys 'encoder' and 'decoder' specifying the number of layers (length of the list)
        and their channel count as a list of int
        :param down_conv: function for the convolution in downsampling layers
        :param up_conv: function for the convolution in upsampling layers
        :param res_block: function that is called for each residual block
        """
        super().__init__()

        self.encoder = []
        self.decoder = []
        self.residuals = self._parse_residuals(n_residual)
        self.input_channels = input_channels
        self.channels = self._parse_channels(channels, n_layers, channel_factor, max_channels)

        # define building blocks
        self.down_conv = down_conv
        self.up_conv = up_conv
        self.res_block = res_block
        self.final_norm = final_norm

        self._build()

    @staticmethod
    def _parse_channels(channels, n_layers, channel_factor, max_channels):
        """
        method for constructing the basic architecture of the network and consolidating the two methods of construction
        mentioned in init
        :param channels: dict with keys 'encoder' and 'decoder' specifying the number of layers (length of the list)
        and their channel count as a list of int
        :param channel_factor: number of channels of the tensor after the inital convolution.
        :param n_layers: number of down- and upsampling layers in the encoder and decoder respectively
        :param max_channels: maximum amount of channels that should be present in the network
        :return: tuple with two lists specifying the number of channels in each layer of the encoder / decoder
        """
        if isinstance(channels, dict):

            encoder_channels = channels.get('encoder', None)
            if encoder_channels is None:
                encoder_channels = ConvAE.calculate_channels(n_layers, channel_factor, max_channels)

            decoder_channels = channels.get('decoder', None)
            if decoder_channels is None:
                n_layers = len(encoder_channels)
                max_channels = encoder_channels[-1]
                channel_factor = max(max_channels // 2 ** (n_layers - 1), 1)
                decoder_channels = ConvAE.calculate_channels(n_layers, channel_factor, max_channels)
                decoder_channels.reverse()

        elif isinstance(channels, Sequence):

            encoder_channels = list(channels)
            decoder_channels = encoder_channels.copy()
            decoder_channels.reverse()

        else:

            encoder_channels = ConvAE.calculate_channels(n_layers, channel_factor, max_channels)
            decoder_channels = encoder_channels.copy()
            decoder_channels.reverse()

        return encoder_channels, decoder_channels

    @staticmethod
    def _parse_residuals(n_residual):
        """
        parse the n_residual argument. If only one tuple is given, residual blocks in the encoder and decoder
        will be symmetric.
        :param n_residual: either a single tuple or sequence of two tuples. the tuples specify the number of residual
        blocks and the number of convolutions per block. If two tuples are passed, the first will be used for the
        encoder and the second for the decoder
        :return: specification of residual blocks for both encoder and decocer
        """

        assert isinstance(n_residual, Sequence), 'n_residual must be a tuple of the form (n_blocks, n_convs_per_block)'

        if not isinstance(n_residual[0], Sequence):
            n_residual = (n_residual, n_residual)

        return n_residual

    @staticmethod
    def calculate_channels(n_layers, channel_factor, max_channels):
        """
        method for calculating the list of channels in each layer based on doubling of channels after each layer
        :param n_layers: number of layers
        :param channel_factor: starting point for the number of channels
        :param max_channels: maximum amount of channels that can be reached
        :return: list of int standing for the number channels
        """
        max_channels = max_channels if max_channels is not None else 2 ** 16
        channels = np.array([channel_factor * 2 ** layer for layer in range(n_layers)])
        channels[1:][channels[1:] > max_channels] = max_channels

        return list(channels)

    def _build(self):
        """
        method that actually builds the network
        :return: None
        """
        # build encoder

        encoder_residuals, decoder_residuals = self.residuals
        encoder_channels, decoder_channels = self.channels

        # initial convolution
        conv = partial(self.down_conv, stride=(1, 1))
        self.encoder.append(conv(self.input_channels, encoder_channels[0]))
        self.add_module('initial_conv', self.encoder[-1])

        channels = zip(encoder_channels[:-1], encoder_channels[1:])
        for depth, (current_channels, out_channels) in enumerate(channels):

            # res blocks
            for res_index in range(encoder_residuals[0]):
                self.encoder.append(self.res_block(channels=current_channels, n_convolutions=encoder_residuals[1]))
                self.add_module('r-block{}-{}'.format(depth + 1, res_index + 1), self.encoder[-1])

            # down-sampling convolution
            self.encoder.append(self.down_conv(current_channels, out_channels))
            self.add_module('conv{}'.format(depth + 1), self.encoder[-1])

        # build decoder

        # invert channel order
        channels = zip(decoder_channels[:-1], decoder_channels[1:])
        n_layers = len(decoder_channels)
        for depth, (current_channels, out_channels) in enumerate(channels):

            # up-sampling convolution
            self.decoder.append(self.up_conv(current_channels, out_channels))
            self.add_module('dconv{}'.format(n_layers - depth - 1), self.decoder[-1])

            # res blocks
            for res_index in range(decoder_residuals[0]):
                self.decoder.append(self.res_block(channels=out_channels, n_convolutions=decoder_residuals[1]))
                self.add_module('dr-block{}-{}'.format(depth + 1, res_index + 1), self.decoder[-1])

        # output convolution
        self.decoder.append(conv(out_channels, self.input_channels))
        self.add_module('output_conv', self.decoder[-1])
        if not self.final_norm:
            self.output_conv.norm = None

    def _forward(self, x, layers=None):
        """
        internally used method for inference. iterates over the specified layers and applies them to the input
        :param x: input tensor
        :param layers: list of callable layers
        :return: output of the inference
        """
        out = x
        layers = layers if layers is not None else self.encoder + self.decoder
        for l in layers:
            out = l(out)

        return out

    def forward(self, x):
        """
        method for performing full forward pass through the network
        :param x: input tensor
        :return: reconstruction of the input tensor
        """
        return self.decode(self.encode(x))

    def encode(self, x):
        """
        method for performing partial inference through the encoder only
        :param x: input tensor
        :return: encoded representation of the input tensor
        """
        return self._forward(x, self.encoder)

    def decode(self, x):
        """
        method for performing partial inference through the decoder only
        :param x: tensor from the representation space
        :return: reconstruction
        """
        return self._forward(x, self.decoder)

