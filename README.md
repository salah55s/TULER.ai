[![DOI](https://zenodo.org/badge/669341284.svg)](https://zenodo.org/badge/latestdoi/669341284)
# TULER.ai

# Description

TULERNN is a novel neural network module developed by TULER, a startup in the field of artificial intelligence. The module introduces innovative features that provide users with enhanced control and flexibility over neural network behavior. With TULERNN, researchers and developers can create sophisticated neural network architectures that cater to a wide range of applications and challenges. It's better than normal NN, with any type of data or with any used activation function, It will reshape all the way we work with AI.

# Key Features

1- Threshold Activation: TULERNN implements a unique threshold activation mechanism for each neuron in the layer. By associating a threshold value with each neuron, it allows users to control neuron activation based on threshold conditions. Neurons can be selectively suppressed or activated, leading to more refined and tailored behavior. This feature is superior to traditional activation functions as it provides fine-grained control over individual neuron responses.

2- Customizable Weight Initialization: The weight parameters in TULERNN are initialized using Xavier uniform initialization, promoting better convergence during training and reducing the risk of vanishing or exploding gradients. This allows users to initialize the model effectively for a wide variety of tasks and architectures.

3- Optional Bias Term: TULERNN supports an optional bias term that can be added to the output of each neuron. The bias term further augments the customization possibilities of the layer, enabling users to fine-tune model behavior.

4- Versatility with Any Data: TULERNN can be seamlessly integrated into neural network architectures designed for any type of data, including text, images, audio, and more. Its adaptability makes it suitable for a wide range of applications, from natural language processing to computer vision and beyond.

5- Compatibility with Any Activation Function: TULERNN works harmoniously with various activation functions, allowing researchers and developers to explore different non-linearities and choose the most suitable one for their specific tasks.

6- Reshaping AI Workflows: TULERNN reshapes the way we approach AI model design and development. Its innovative features enable users to build highly customized and efficient neural network architectures that were previously challenging or impossible to achieve.


# BrainNeuronActivation - Brain-Inspired Neural Activation Module

The BrainNeuronActivation module is a neural network component designed to mimic the behavior of neurons in the brain. Inspired by the intricate functionality of biological neurons, this module introduces parameters that control the activation behavior of artificial neurons, making it more biologically plausible.

Features:

1- Brain-Inspired Activation
The BrainNeuronActivation module introduces unique activation behaviors to artificial neurons. It includes the following key features:

2- Thresholding Mechanism: Each neuron is associated with a threshold value. If the neuron's output is below this threshold, it is suppressed, simulating the concept of resting potential in biological neurons.

3- Customizable Activation: The module allows you to customize the behavior of neurons based on input. When the input is negative, neurons can be suppressed with a certain parameter (parm1). Depending on the input's magnitude, neurons can fire an action potential (parm5 + parm2 * (output - parm4)) or simply linearly scale the output (parm3 * output).

# Usage:
First, install the file in your "....\Lib\site-packages\torch\nn\modules\" 
```
from torch.nn.modules.tulerai import TULERNN ,BrainNeuronActivation

tuler_nn = TULERNN(in_features=100, out_features=50)

num_neurons = 50  # Replace this with the desired number of neurons

brain_activation = BrainNeuronActivation(num_neurons=num_neurons)

#Or
class Try(torch.nn.Module):
    def __init__(self, input_features, output_features, dropout=0.7):
        super(Try, self).__init__()
        self.output_features = output_features
        self.network= torch.nn.Sequential(
            TULERNN(input_features, 5120),
            BrainNeuronActivation(5120),
            torch.nn.Dropout(dropout),
            TULERNN(5120, output_features),
            torch.nn.Sigmoid()
        )
    def forward(self, inputs):
        output=self.network(inputs)
        return output
```

# Feedback and Contributions
TULER welcomes feedback, bug reports, and contributions to enhance the TULERNN module further. If you encounter any issues or have ideas for improvements, please feel free to open an issue or create a pull request on this GitHub repository. Make sure to cite this work, and share your results in a pull request, As it will be documented in a paper.
