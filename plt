import sys
sys.path.append('../')
from pycore.tikzeng import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Input
    to_input("input.jpg"),

    # Conv1 + LeakyReLU
    to_Conv("conv1", s_filer=1, n_filer=64, offset="(0,0,0)", to="(0,0,0)", height=40, depth=40, width=2, caption="Conv1d 64\nLeakyReLU"),

    # Conv2 + BatchNorm + LeakyReLU
    to_Conv("conv2", s_filer=64, n_filer=128, offset="(2,0,0)", to="(conv1-east)", height=32, depth=32, width=2, caption="Conv1d 128\nBatchNorm\nLeakyReLU"),
    to_connection("conv1", "conv2"),

    # Conv3 + BatchNorm + LeakyReLU
    to_Conv("conv3", s_filer=128, n_filer=256, offset="(2,0,0)", to="(conv2-east)", height=16, depth=16, width=2, caption="Conv1d 256\nBatchNorm\nLeakyReLU"),
    to_connection("conv2", "conv3"),

    # Conv4 + BatchNorm + LeakyReLU
    to_Conv("conv4", s_filer=256, n_filer=512, offset="(2,0,0)", to="(conv3-east)", height=8, depth=8, width=2, caption="Conv1d 512\nBatchNorm\nLeakyReLU"),
    to_connection("conv3", "conv4"),

    # Final Conv
    to_Conv("conv5", s_filer=512, n_filer=1, offset="(2,0,0)", to="(conv4-east)", height=4, depth=4, width=2, caption="Conv1d 1\nKernel=17"),
    to_connection("conv4", "conv5"),

    to_end()
]

def main():
    to_generate(arch, "discriminator.tex")

if __name__ == '__main__':
    main()


import sys
sys.path.append('../')
from pycore.tikzeng import *

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Latent vector
    to_input("latent.jpg"),

    # Deconv1 + BatchNorm + ReLU
    to_Conv("deconv1", s_filer=100, n_filer=512, offset="(0,0,0)", to="(0,0,0)", height=8, depth=8, width=2, caption="ConvT1d 512\nBatchNorm\nReLU"),

    # Deconv2 + BatchNorm + ReLU
    to_Conv("deconv2", s_filer=512, n_filer=256, offset="(2,0,0)", to="(deconv1-east)", height=16, depth=16, width=2, caption="ConvT1d 256\nBatchNorm\nReLU"),
    to_connection("deconv1", "deconv2"),

    # Deconv3 + BatchNorm + ReLU
    to_Conv("deconv3", s_filer=256, n_filer=128, offset="(2,0,0)", to="(deconv2-east)", height=32, depth=32, width=2, caption="ConvT1d 128\nBatchNorm\nReLU"),
    to_connection("deconv2", "deconv3"),

    # Deconv4 + BatchNorm + ReLU
    to_Conv("deconv4", s_filer=128, n_filer=64, offset="(2,0,0)", to="(deconv3-east)", height=64, depth=64, width=2, caption="ConvT1d 64\nBatchNorm\nReLU"),
    to_connection("deconv3", "deconv4"),

    # Output layer
    to_Conv("output", s_filer=64, n_filer=1, offset="(2,0,0)", to="(deconv4-east)", height=128, depth=128, width=2, caption="ConvT1d 1\nTanh"),
    to_connection("deconv4", "output"),

    to_end()
]

def main():
    to_generate(arch, "generator.tex")

if __name__ == '__main__':
    main()

def to_cor():
    return r"""
\def\ConvColor{rgb:blue,5;cyan,2.5;white,5}       % couleur principale des conv
\def\ConvReluColor{rgb:blue,5;cyan,5;white,5}     % couleur des bandes ReLU
\def\PoolColor{rgb:blue,3;black,0.3}             % couleur des pools
\def\UnpoolColor{rgb:blue,4;green,1;black,0.3}   % couleur des unpool
\def\FcColor{rgb:blue,5;cyan,2.5;white,5}       % si tu avais FC
\def\FcReluColor{rgb:blue,5;cyan,5;white,4}     
\def\SoftmaxColor{rgb:blue,5;black,7}           % softmax en bleu
\def\SumColor{rgb:blue,5;green,15}              % somme / addition en bleu
"""