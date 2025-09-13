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

    def forward(self, x: torch.Tensor):
        residual_connection = x

        x = self.groupnorm_1(x)
        x = F.relu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.relu(x)
        x = self.conv_2(x)

        x = x + self.residual_connection(residual_connection)

        return x


if __name__ == "__main__":
    rb = ResidualBlock(8, 16, 4).cuda()
    tn = torch.randn(3, 8, 32, 32, device="cuda")
    out = rb(tn)
    print(out.shape)
