import torch
from torch import nn

# DCGAN
class Discriminator(nn.Module):
    def __init__(self, img_shape, feature_dim, dim_factor):
        super(Discriminator, self).__init__()

        C, W, H = img_shape

        self.feature_dim = feature_dim
        self.dim_factor = dim_factor

        self.middle = nn.ModuleList()
        
        self.conv_in = nn.Sequential(
            nn.Conv2d(C, self.feature_dim * dim_factor[0], 4, 2, 1, bias=False),
            # nn.BatchNorm2d(self.feature_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        size = W // 2

        last_feature_dim = self.feature_dim
        for factor in self.dim_factor[1:]:
            next_feature_dim = self.feature_dim * factor
            self.middle.append(
                nn.Sequential(
                    nn.Conv2d(last_feature_dim, next_feature_dim, 4, 2, 1, bias=False),
                    # nn.BatchNorm2d(next_feature_dim),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            last_feature_dim = next_feature_dim
            size = size // 2

        self.conv_out = nn.Sequential(
            nn.Conv2d(last_feature_dim, 1, size, 1, 0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, img):
        x = self.conv_in(img)
        for layer in self.middle:
            x = layer(x)
        x = self.conv_out(x)
        return x

class Generator(nn.Module):
    def __init__(self, z_dim, img_shape, feature_dim, dim_factor) -> None:
        super(Generator, self).__init__()
        C, H, W = img_shape
        last_size = H
        size_list = [last_size]
        for _ in dim_factor:
            last_size = last_size // 2
            size_list.append(last_size)
        self.conv_in = nn.Sequential(nn.ConvTranspose2d(z_dim, feature_dim*dim_factor[-1], last_size, bias=False),
                                        # nn.BatchNorm2d(feature_dim*dim_factor[-1]),
                                        nn.ReLU(inplace=True))
        self.middle = nn.ModuleList()
        last_feature_dim = feature_dim*dim_factor[-1]
        dim_factor.reverse()
        size_list.reverse()
        for factor, size in zip(dim_factor[1:], size_list[1:]):
            next_feature_dim = feature_dim*factor
            if size % 2 == 0:
                self.middle.append(nn.Sequential(nn.ConvTranspose2d(last_feature_dim, next_feature_dim, 4, 2, 1, bias=False),
                                                    # nn.BatchNorm2d(next_feature_dim),
                                                    nn.ReLU(inplace=True)))
            else:
                self.middle.append(nn.Sequential(nn.ConvTranspose2d(last_feature_dim, next_feature_dim, 4, 2, 1, output_padding=1, bias=False),
                                                    # nn.BatchNorm2d(next_feature_dim),
                                                    nn.ReLU(inplace=True)))
            last_feature_dim = next_feature_dim
        self.conv_out = nn.Sequential(nn.ConvTranspose2d(last_feature_dim, C, 4, 2, 1, bias=False),
                                      nn.Tanh())

    def forward(self, z):
        x = self.conv_in(z)
        for layer in self.middle:
            x = layer(x)
        x = self.conv_out(x)
        return x


if __name__ == '__main__':
    img_shape = (3, 28, 28)
    feature_dim = 64
    dim_factor = [1, 2, 4]

    D = Discriminator(img_shape, feature_dim, dim_factor)
    img = torch.randn(1, *img_shape)
    print(D(img))

    z_dim = 100
    G = Generator(z_dim, img_shape, feature_dim, dim_factor)
    z = torch.randn(1, z_dim, 1, 1)
    print(G)
    print(G(z).shape)