import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, groupnorm_num_groups: int):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(groupnorm_num_groups, in_channels)
        self.conv_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding="same"
        )

        self.groupnorm_2 = nn.GroupNorm(groupnorm_num_groups, out_channels)
        self.conv_2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding="same"
        )

        self.residual_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_connection = x

        x = self.groupnorm_1(x)
        x = F.relu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.relu(x)
        x = self.conv_2(x)

        x = x + self.residual_connection(residual_connection)

        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, interpolate: bool = False):
        super().__init__()

        if interpolate:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding="same"
                ),
            )

        else:
            self.upsample = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.upsample(inputs)


if __name__ == "__main__":
    # m = ResidualBlock(8, 16, 4).cuda()
    m = UpsampleBlock(8, 16).cuda()
    tn = torch.randn(3, 8, 32, 32, device="cuda")
    out = m(tn)
    print(out.shape)
