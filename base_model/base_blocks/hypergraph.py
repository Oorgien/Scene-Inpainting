import torch
from torch import nn
from torch.nn import functional as F

from utils import same_padding


class HypergraphConv(nn.Module):
    def __init__(
            self,
            in_channels, out_channels,
            filters, edges,
            height, width
    ):
        """
        :param in_channels:
        :param out_channels:
        :param filters: Intermeditate channels for phi and lambda matrices - A Hyperparameter
        :param edges: hypergraph edges
        :param height: height of input tensor
        :param width: width of input tensor
        """

        super(HypergraphConv, self).__init__()
        self.filters = filters
        self.edges = edges
        self.lambda_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=[1, 1],
            stride=[1, 1]
        )

        self.psi_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=[1, 1],
            stride=[1, 1]
        )

        self.omega_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=edges,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=same_padding(
                height, width,
                [3, 3], [1, 1], [1, 1]
            )
        )

        self.out_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[7, 7],
            stride=[1, 1],
            padding=same_padding(
                height, width,
                [7, 7], [1, 1], [1, 1]
            )
        )

    def forward(self, x):
        """
        :param x: input tensor (batch_size x channels x width x height)

        H = psi * lambda * psi.T * omega
        out = D^0.5 * H * B^0.5 * H^0.5.T * D^0.5.T * X + bias
        :return:
        """

        width, height = x.shape[2], x.shape[3]
        vertices = width * height

        # Lambda
        Lambda = F.avg_pool2d(x, max(width, height), stride=1)  # shape [N, C, 1, 1]
        Lambda = self.lambda_conv(Lambda).squeeze().squeeze()
        Lambda = torch.diag_embed(Lambda)  # shape [N, C, C]

        # Psi
        Psi = self.psi_conv(x).permute(0, 2, 3, 1)  # shape [N, w, h, C]
        Psi = Psi.reshape(-1, vertices, self.filters)  # shape [N, vertices, C]

        # Omega
        Omega = self.omega_conv(x).permute(0, 2, 3, 1)  # shape [N, w, h, edges]
        Omega = Omega.reshape(-1, vertices, self.edges)  # shape [N, vertices, edges]

        # Weighted incidence matrix
        # H = Psi * Lambda * Psi.T * Omega
        H = torch.matmul(torch.matmul(torch.matmul(Psi, Lambda), torch.transpose(Psi, 1, 2)), Omega)
        H = torch.abs(H)
        # Out shape: [N, vertices, edges]

        # Degree matrix
        # How many edges go through the vertices
        D = torch.sum(H, dim=2)  # shape [N, vertices]

        # Edge degree matrix
        # How many vertices lie on the edge
        B = torch.sum(H, dim=1)  # shape [N, edges]
        B = torch.diag_embed(torch.pow(B, -1))

        # Mutlpying with the incidence matrix to ensure no matrix developed is of large size - (vertices * vertices)
        D_H = torch.matmul(torch.diag_embed(torch.pow(D, -0.5)), H)  # out shape: [N, vertices, edges]

        features = x.permute(0, 2, 3, 1)
        features = torch.reshape(features, (-1, vertices, x.shape[1]))

        out = features - torch.matmul(D_H, torch.matmul(B, torch.matmul(torch.transpose(D_H, 1, 2), features)))
        out = out.permute(0, 2, 1)
        out = out.reshape(-1, out.shape[1], width, height)
        out = self.out_conv(out)
        return out


def test_hypergrashconv(device_id):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    img = torch.rand(size=(10, 64, 64, 64)).to(device)
    HGCV = HypergraphConv(
        in_channels=64, out_channels=128,
        filters=64, edges=256,
        height=64, width=64
    ).to(device)

    out = HGCV(img)
    assert out.shape == (10, 128, 64, 64)
    return True
