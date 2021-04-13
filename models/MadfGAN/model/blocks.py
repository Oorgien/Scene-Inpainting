import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.nn import functional as F

from base_model.base_blocks import _activation, _norm, _padding, conv_block, upconv_block


class MADF(nn.Module):
    def __init__(self,
                 in_channels_m, out_channels_m,
                 in_channels_e, out_channels_e,
                 kernel_size_m, kernel_size_e,
                 stride_m, stride_e,
                 padding_m, padding_e,
                 activation_m='relu', activation_e='relu',
                 norm_m='none', norm_e='bn'):
        """
        :param in_channels_m: input channels of mask layer - m {l-1}
        :param out_channels_m: output channels of mask layer - m {l}
        :param in_channels_e: input chanels of image layer - e {l-1}
        :param out_channels_e: output channels of image layer - e {l}

        :param kernel_size_m: kernel size for transformation m {l-1} -> m {l}
        :param kernel_size_e: kernel size for transformation e {l-1} -> e {l}
            and for kernel creation m {l} -> kernel for e {l} convolution

        :param stride_m: stride for m {l-1} -> m {l} transformation
        :param stride_e: stride for e {l-1} -> e {l} transformation

        :param padding_m: padding for m {l-1} -> m {l} transformation
        :param padding_e: padding for e {l-1} -> e {l} transformation

        :param activation_m: activation_m for m {l-1} -> m {l} transformation
        :param activation_e: activation_e for e {l-1} -> e {l} transformation
        """
        super(MADF, self).__init__()
        self.in_channels_m = in_channels_m
        self.out_channels_m = out_channels_m
        self.in_channels_e = in_channels_e
        self.out_channels_e = out_channels_e

        self.kernel_size_e = kernel_size_e
        self.kernel_size_m = kernel_size_m

        self.padding_e = padding_e
        self.padding_m = padding_m

        self.stride_e = stride_e
        self.stride_m = stride_m

        self.activation_m = activation_m
        self.activation_e = activation_e

        self.norm_m = norm_m
        self.norm_e = norm_e

        self.conv_m = conv_block(
            in_channels=in_channels_m,
            out_channels=out_channels_m,
            kernel_size=kernel_size_m,
            stride=stride_m,
            padding=padding_m,
            activation=activation_m,
            norm=norm_m)

        self.conv_filters = conv_block(
            in_channels=out_channels_m,
            out_channels=in_channels_e * kernel_size_e *
            out_channels_e * kernel_size_e,
            kernel_size=1,
            stride=1,
            padding=0,
            activation="none",
            norm='none')

    def forward(self, m_l_1, e_l_1):
        """
        :param m_l_1: mask input layer
        :param e_l_1: image input layer
        :return: m {l} and e {l}
        """

        # Create m {l} layer
        m_l = self.conv_m(m_l_1)

        # Check width and height of m {l} to count convolution windows for e {l-1} conv
        N_h, N_w = m_l.shape[2], m_l.shape[3]

        # Applying 1 x 1 convolution to form convolution e {l-1} -> e {l} kernel matrix
        filter = self.conv_filters(m_l).reshape(
            m_l.shape[0], N_h, N_w, self.out_channels_e,
            self.in_channels_e, self.kernel_size_e, self.kernel_size_e)

        # Padding for e {l-1} manually because for each { kernel_size_e x kernel_size_e }
        # Window we compute convolution with it's own convolution kernel
        pad = [self.padding_e] * 4
        e_l_1 = F.pad(e_l_1, pad, "constant", 0)

        # Initializing kernel matrix to store
        e_l = torch.zeros((m_l_1.shape[0], self.out_channels_e, N_h, N_w))
        for batch_id in range(m_l.shape[0]):
            for i in range(0, filter.shape[0], self.stride_e):
                for j in range(0, filter.shape[1], self.stride_e):
                    product = F.conv2d(
                        e_l_1[batch_id, :, i:i + self.kernel_size_e, j:j + self.kernel_size_e].unsqueeze(0),
                        filter[batch_id, i, j, :, :, :, :].squeeze(0).squeeze(0))
                    e_l[batch_id, :, i, j] = product.view(1, self.out_channels_e)

        activation_e = _activation(self.activation_e)
        norm = _norm(self.norm_e, self.out_channels_e)
        if activation_e:
            e_l = activation_e(e_l)
        if norm:
            e_l = norm(e_l)
        return m_l, e_l


class RecovecyBlock(nn.Module):
    def __init__(self,
                 in_channels_r, out_channels_r, in_channels_u, out_channels,
                 kernel_size_in, kernel_size_out,
                 up_stride_in, stride_out,
                 up_padding_in, padding_out, output_padding=0,
                 activation_in='lrelu', activation_out='lrelu',
                 norm_in='bn', norm_out='none'):
        """
        u {l-1} and r {l} new should be of equal sizes to be concatenated

        :param in_channels_r: in channels for r {l}
        :param out_channels_r: out channels for r {l-1}
        :param in_channels_u: in channels for u {l-1}
        :param out_channels: out channels for (u {l-1}, r {l}) -> r {l-1}

        :param kernel_size_in: kernel size for r {l}) -> r {l} new
        :param kernel_size_out: kernel size for (u {l-1}, r {l}) -> r {l-1}

        :param up_stride_in: stride for transposed conv r {l}) -> r {l} new
        :param stride_out: stride  for (u {l-1}, r {l}) -> r {l-1}

        :param up_padding_in: padding for transposed convolution r {l} -> r {l} new
        :param output_padding: output padding for transposed convolution r {l} -> r {l} new
        :param padding_out: padding for (u {l-1}, r {l}) -> r {l-1}

        :param activation_in: activation layer for r {l}) -> r {l} new
        :param activation_out: activation layer for (u {l-1}, r {l}) -> r {l-1}

        :param norm_in: normalization layer for r {l}) -> r {l} new
        :param norm_out: normalization layer for (u {l-1}, r {l}) -> r {l-1}
        """
        super(RecovecyBlock, self).__init__()

        self.in_upconv = upconv_block(
            in_channels = in_channels_r,
            out_channels = out_channels_r,
            kernel_size=kernel_size_in,
            stride=up_stride_in,
            padding=up_padding_in,
            output_padding = output_padding,
            norm=norm_in,
            activation=activation_in
        )

        self.out_conv = conv_block(
            in_channels=out_channels_r + in_channels_u,
            out_channels=out_channels,
            kernel_size=kernel_size_out,
            stride=stride_out,
            padding=padding_out,
            norm=norm_out,
            activation=activation_out
        )

    def forward(self, r_l, u_l_1):
        r_l_new = self.in_upconv(r_l)
        r_l_1 = self.out_conv(torch.cat((r_l_new, u_l_1), dim=1))
        return r_l_1
    

class RefinementBlock(nn.Module):
    def __init__(self,
                 in_channels_1, in_channels_2, out_channels,
                 kernel_size_1, kernel_size_2,
                 stride_1, up_stride_2,
                 padding_1, up_padding_2, output_padding=0,
                 activation_in='relu', activation_out='lrelu',
                 norm_in='bn', norm_out='none'):
        """
        1 convolution - from f {l-1}{r-1}
        2 transposed convolution - from f {l} {k}
        3 convolution - the one to be summed with result of f {l}{k} (2) convolution
        4 convolution - the one to be producted with result of f {l}{k} (2) convolution

        :param in_channels_1: Input channels of the 1st convolution layer
        :param in_channels_2: Input channels of the 2st convolution layer
        :param out_channels: Output channels of the 1st, 2st, 3rd, 4th convolution layers

        :param kernel_size_1: Kernel size of 1st convolution layer
        :param kernel_size_2: Kernel size of 2nd transposed convolution layer

        :param stride_1: Stride of 1st convolution layer
        :param up_stride_2: Stride of 2nd transposed convolution layer

        :param padding_1: Padding of 1st convolution layer
        :param up_padding_2: Padding of 2nd transposed convolution layer
        :param output_padding: Output padding of 2nd transposed convolution layer

        :param activation_in: Activation layer of 1st convolution layer
        :param activation_out: Activation layer 2nd transposed convolution layer

        :param norm_in: Normalization layer of 1st convolution layer
        :param norm_out: Normalization layer of 2nd transposed convolution layer
        """

        super(RefinementBlock, self).__init__()

        self.conv_1 = conv_block(
            in_channels=in_channels_1,
            out_channels=out_channels,
            kernel_size=kernel_size_1,
            stride=stride_1,
            padding=padding_1,
            norm='none',
            activation=activation_in
        )

        self.upconv_2 = upconv_block(
            in_channels = in_channels_2,
            out_channels = out_channels,
            kernel_size = kernel_size_2,
            stride = up_stride_2,
            padding = up_padding_2,
            output_padding = output_padding,
            norm=norm_in,
            activation='none'
        )

        self.conv_3 = conv_block(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm='none',
            activation='none'
        )

        self.conv_4 = conv_block(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm='none',
            activation='none'
        )

        self.out_act = _activation(act_type=activation_out)
        self.out_norm = _norm(norm_type=norm_out, channels=out_channels)

    def forward(self, f_1, f_2):
        # The one from recovery decoder
        f_1 = self.conv_1(f_1)

        # The one from l layer
        f_2 = self.upconv_2(f_2)

        # The one to be summed up with product
        f_3 = self.conv_3(f_1)

        # The one to be multiplied
        f_4 = self.conv_4(f_1)

        product = f_2 * f_4
        sum = product + f_3

        if self.out_norm:
            sum = self.out_norm(sum)
        if self.out_act:
            sum = self.out_act(sum)
        return sum


def test_blocks():
    print("MADF test...")
    block = MADF(
        in_channels_m=3, out_channels_m=4,
        in_channels_e=3, out_channels_e=4,
        kernel_size_m=3, kernel_size_e=3,
        stride_m=2, stride_e=2,
        padding_m=1, padding_e=1,
        activation_m='none', activation_e='relu',
        norm_m='none', norm_e='in'
    )

    m_l_1 = torch.randn(size=(10, 3, 64, 64))
    e_l_1 = torch.randn(size=(10, 3, 64, 64))
    m_l, e_l = block.forward(m_l_1, e_l_1)
    assert (m_l.shape == (10, 4, 32, 32))
    assert (e_l.shape == (10, 4, 32, 32))
    print("MADF test passed --- OK")

    print("Recovery block test...")
    u_l_1 = torch.randn(size=(10, 6, 64, 64))
    r_l = torch.randn(size=(10, 3, 32, 32))

    block = RecovecyBlock(
        in_channels_r=3, out_channels_r=6,
        in_channels_u=6, out_channels=32,
        kernel_size_in=3, kernel_size_out=3,
        up_stride_in=2, stride_out=1,
        up_padding_in=1, padding_out=1,
        output_padding=1,
        activation_in='lrelu', activation_out='lrelu',
        norm_in='bn', norm_out='none'
    )

    r_l_1 = block.forward(r_l, u_l_1)
    assert (r_l_1.shape == (10, 32, 64, 64))
    print("Recovery block test --- OK")

    print("Refinement Block test...")
    f_2 = torch.randn(size=(10, 16, 32, 32))
    f_1 = torch.rand(size=(10, 8, 64, 64))
    block = RefinementBlock(
        in_channels_1=8,
        in_channels_2=16,
        out_channels=32,
        kernel_size_1=3,
        kernel_size_2=3,
        stride_1=1,
        up_stride_2=2,
        padding_1=1,
        up_padding_2=1,
        output_padding=1,
        activation_in='relu', activation_out='lrelu',
        norm_in='bn', norm_out='none')

    out = block.forward(f_1, f_2)
    assert (out.shape == (10, 32, 64, 64))
    print("Refinement Block test --- OK")


if __name__ == "__main__":
    test_blocks()
