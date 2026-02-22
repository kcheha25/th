import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pycore.tikzeng import *
from pycore.blocks import *
import sys

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Latent vector
    to_FC("latent", 100, caption="Latent z"),

    # Deconv1
    to_Conv("deconv1", 4, 32, 512,
            caption="ConvT1d 512\nReLU"),
    to_FC("bn1", 20, caption="BatchNorm1d"),

    # Deconv2
    to_Conv("deconv2", 8, 32, 256,
            caption="ConvT1d 256\nReLU"),
    to_FC("bn2", 20, caption="BatchNorm1d"),

    # Deconv3
    to_Conv("deconv3", 16, 32, 128,
            caption="ConvT1d 128\nReLU"),
    to_FC("bn3", 20, caption="BatchNorm1d"),

    # Deconv4
    to_Conv("deconv4", 32, 32, 64,
            caption="ConvT1d 64\nReLU"),
    to_FC("bn4", 20, caption="BatchNorm1d"),

    # Output
    to_FC("output", 1,
          caption="ConvT1d 1\nTanh"),

    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile)

if __name__ == '__main__':
    main()


from pycore.tikzeng import *
from pycore.blocks import *
import sys

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),

    # Input
    to_input("input.jpg"),

    # Conv1
    to_Conv("conv1", 64, 32, 64,
            caption="Conv1d 64\nLeakyReLU"),
    
    # Conv2
    to_Conv("conv2", 32, 32, 128,
            caption="Conv1d 128\nLeakyReLU"),
    to_FC("bn2", 20, caption="BatchNorm1d"),

    # Conv3
    to_Conv("conv3", 16, 32, 256,
            caption="Conv1d 256\nLeakyReLU"),
    to_FC("bn3", 20, caption="BatchNorm1d"),

    # Conv4
    to_Conv("conv4", 8, 32, 512,
            caption="Conv1d 512\nLeakyReLU"),
    to_FC("bn4", 20, caption="BatchNorm1d"),

    # Final Conv
    to_FC("conv_final", 1,
          caption="Conv1d 1\nKernel=17"),

    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile)

if __name__ == '__main__':
    main()