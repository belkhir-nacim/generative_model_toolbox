from generative_models_toolbox.layers import ae as parts
from .base import ConvAE
import numpy as np
import torch as pt
import torch.nn as nn
from torch.nn.functional import relu
from collections import Sequence
from functools import partial



class ConvAE2d(ConvAE):
    """
    Class representing a convolutional autoencoder for 2d inputs. The architecture of the AE is highly customizable with
    regards to the number of layers, kernel size, stride in downsampling, functions for convolution, padding and
    normalization among other things. In this sense it is an attempt at a semi-general constructor for an AE.
    """

    def __init__(self, channel_factor=None, n_layers=None, activation=relu, kernel_size=(4, 4), stride=(2, 2),
                 n_residual=(0, 0), max_channels=None, input_channels=1, affine=False, channels=None,
                 padding=pt.nn.ReflectionPad2d, norm=pt.nn.InstanceNorm2d,
                 down_conv=pt.nn.Conv2d, up_conv=parts.ConvResize2d, final_norm=True, **kwargs):
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
        :param channel_factor: number of channels of the tensor after the inital convolution.
        :param n_layers: number of down- and upsampling layers in the encoder and decoder respectively
        :param activation: activation function that will be applied after each convolution
        :param kernel_size: kernel size of the convolutions
        :param stride: stride of the down- and upsampling layers
        :param n_residual: tuple that controls the residual blocks in the network. The first entry specifies the amount
        of residual blocks in between down- or upsampling layers. The second specifies the amount of convolutions in
        each block. Alternatively, two tuples can be passed. The first will be used for the encoder and the second for
        the decoder.
        :param max_channels: maximum amount of channels that should be present in the network
        :param input_channels: number of channels in the input tensor
        :param affine: boolean indicating whether the normalization uses bias or not
        :param channels: dict with keys 'encoder' and 'decoder' specifying the number of layers (length of the list)
        and their channel count as a list of int
        :param padding: function that is used for padding the tensors before convolving, None for no padding
        :param norm: function for normalizing the tensors after convolution, None for no normalization
        :param down_conv: function for the convolution in downsampling layers and residual blocks
        :param up_conv: function for the convolution in upsampling layers
        :param kwargs: additional keyword arguments that can be passed to the down_conv function
        """

        # define building blocks
        res_block = partial(parts.ResBlock2d, kernel_size=kernel_size,
                            convolution=down_conv, norm=norm, activation=activation,
                            affine=affine, padding=padding, **kwargs)

        up_conv = partial(up_conv, kernel_size=kernel_size, upsampling=(stride, 'bilinear'),
                          stride=(1, 1), activation=activation, padding=padding, norm=norm,
                          convolution=down_conv, affine=affine)

        down_conv = partial(parts.GeneralConvolution, kernel_size=kernel_size, stride=stride,
                            activation=activation, padding=padding, norm=norm, convolution=down_conv,
                            affine=affine, **kwargs)
        # up_conv = partial(parts.GeneralConvolution, kernel_size=(4, 4), stride=stride,
        #                          activation=activation, padding=None, norm=norm,
        #                          convolution=partial(pt.nn.ConvTranspose2d, padding=1),
        #                          affine=affine, **kwargs)

        super().__init__(n_layers, n_residual, channel_factor, max_channels, input_channels, channels,
                         down_conv, up_conv, res_block, final_norm)


class ConvAE3d(ConvAE):
    """
    Class representing a convolutional autoencoder for 2d inputs. The architecture of the AE is highly customizable with
    regards to the number of layers, kernel size, stride in downsampling, functions for convolution, padding and
    normalization among other things. In this sense it is an attempt at a semi-general constructor for an AE.
    """

    def __init__(self, channel_factor=None, n_layers=None, activation=relu, kernel_size=(9, 3, 3), stride=(1, 2, 2),
                 n_residual=(0, 0), max_channels=None, input_channels=1, affine=False, channels=None,
                 padding=pt.nn.ReplicationPad3d, norm=pt.nn.InstanceNorm3d,
                 down_conv=pt.nn.Conv3d, up_conv=parts.ConvResize3d, final_norm=True, **kwargs):
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
        :param channel_factor: number of channels of the tensor after the inital convolution.
        :param n_layers: number of down- and upsampling layers in the encoder and decoder respectively
        :param activation: activation function that will be applied after each convolution
        :param kernel_size: kernel size of the convolutions
        :param stride: stride of the down- and upsampling layers
        :param n_residual: tuple that controls the residual blocks in the network. The first entry specifies the amount
        of residual blocks in between down- or upsampling layers. The second specifies the amount of convolutions in
        each block. Alternatively, two tuples can be passed. The first will be used for the encoder and the second for
        the decoder.
        :param max_channels: maximum amount of channels that should be present in the network
        :param input_channels: number of channels in the input tensor
        :param affine: boolean indicating whether the normalization uses bias or not
        :param channels: dict with keys 'encoder' and 'decoder' specifying the number of layers (length of the list)
        and their channel count as a list of int
        :param padding: function that is used for padding the tensors before convolving, None for no padding
        :param norm: function for normalizing the tensors after convolution, None for no normalization
        :param down_conv: function for the convolution in downsampling layers and residual blocks
        :param up_conv: function for the convolution in upsampling layers
        :param kwargs: additional keyword arguments that can be passed to the down_conv function
        """

        # define building blocks
        down_conv = partial(parts.GeneralConvolution, kernel_size=kernel_size, stride=stride,
                            activation=activation, padding=padding, norm=norm, convolution=down_conv,
                            affine=affine, **kwargs)
        up_conv = partial(up_conv, kernel_size=kernel_size, upsampling=(stride, 'trilinear'),
                          stride=(1, 1), activation=activation, padding=padding, norm=norm,
                          convolution=down_conv, affine=affine)
        res_block = partial(parts.ResBlock3d, n_convolutions=n_residual[1], kernel_size=kernel_size,
                            convolution=down_conv, norm=norm, activation=activation,
                            affine=affine, padding=padding, **kwargs)

        super().__init__(n_layers, n_residual, channel_factor, max_channels, input_channels, channels,
                         down_conv, up_conv, res_block, final_norm)