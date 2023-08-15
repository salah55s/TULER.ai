import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torch import Tensor
import torch.nn.init as init
import math
import torch.nn.functional as F
class TULERNN(nn.Module):
    __constants__ = ['in_features', 'out_features']
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TULERNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.threshold = nn.Parameter(torch.empty((out_features, 1), **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.weight)
        init.xavier_uniform_(self.threshold)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    def forward(self, input: Tensor) -> Tensor:
        linear_output = F.linear(input, self.weight, self.bias)
        exceed_threshold = linear_output <=  self.threshold.transpose(0, 1)
        output = linear_output * exceed_threshold + linear_output * (~exceed_threshold)
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class BrainNeuronActivation(nn.Module):
    def __init__(self):
        super(BrainNeuronActivation, self).__init__()

    def forward(self, input, num_neurons=None):
        if num_neurons is None:
            num_neurons = input.size(1)  # Get the number of neurons from the input tensor
        
        thresholds = nn.Parameter(torch.rand(num_neurons))
        parm1 = nn.Parameter(torch.rand(num_neurons))
        parm2 = nn.Parameter(torch.rand(num_neurons))
        parm3 = nn.Parameter(torch.rand(num_neurons))
        parm4 = nn.Parameter(torch.rand(num_neurons))
        parm5 = nn.Parameter(torch.rand(num_neurons))

        output = input.clone()
        output = torch.where(output < 0, parm1 * output, output)
        action_potentials = torch.where(output >= thresholds, parm5 + parm2 * (output - parm4), parm3 * output)
        return action_potentials




class Try_TULER(nn.Module):
    def __init__(self, input_features, output_features, dropout=0.7):
        super(Try_TULER, self).__init__()
        self.output_features = output_features
        self.network = nn.Sequential( 
            TULERNN(input_features,3072),
            BrainNeuronActivation(3072),
            nn.Dropout(dropout),
            TULERNN(3072, out_features=output_features),
            nn.Sigmoid()
        )
    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), -1)  # Reshape to (batch_size, 784)
        output = self.network(inputs)
        return output

    
# Define the Try model
class Try(nn.Module):
    def __init__(self, input_features, output_features, dropout=0.7):
        super(Try, self).__init__()
        self.output_features = output_features
        self.network = nn.Sequential(
            nn.Linear(input_features, 3072),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(3072, output_features),
            nn.Sigmoid()
        )
    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), -1)
        output = self.network(inputs)
        return output
    
    

def train_and_evaluate(model, train_loader, test_loader, lr=0.0001, epochs=200):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Move the model to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Lists to store training and testing accuracies and losses
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        correct_train = 0
        total_train = 0
        train_loss = 0
        with torch.no_grad():
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_train += targets.size(0)
                correct_train += (predicted == targets).sum().item()
                train_loss += criterion(outputs, targets).item()

        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        correct_test = 0
        total_test = 0
        test_loss = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total_test += targets.size(0)
                correct_test += (predicted == targets).sum().item()
                test_loss += criterion(outputs, targets).item()

        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)
        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Accuracy: {train_accuracy:.4f}%, Test Accuracy: {test_accuracy:.4f}%")

    return train_accuracies, train_losses, test_accuracies, test_losses

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

mnist_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
train_size = int(0.8 * len(mnist_dataset))
test_size = len(mnist_dataset) - train_size
train_dataset, test_dataset = random_split(mnist_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Test the Try_TULER model with different activation functions
activation_functions = [
    BrainNeuronActivation(3072),
    nn.ReLU(),
    nn.PReLU(),
]

activation_scores_try_tuler = []
activation_losses_try_tuler = []

for activation_function in activation_functions:
    print(f"Testing Try_TULER with activation: {activation_function.__class__.__name__}")
    model = Try_TULER(input_features=28*28, output_features=10)
    model.network[1] = activation_function
    train_accuracies, train_losses, test_accuracies, test_losses = train_and_evaluate(model, train_loader, test_loader)
    activation_scores_try_tuler.append((train_accuracies, test_accuracies))
    activation_losses_try_tuler.append((train_losses, test_losses))

# Test the Try model with different activation functions
activation_scores_try = []
activation_losses_try = []

for activation_function in activation_functions:
    print(f"Testing Try with activation: {activation_function.__class__.__name__}")
    model = Try(input_features=28*28, output_features=10)
    model.network[1] = activation_function
    train_accuracies, train_losses, test_accuracies, test_losses = train_and_evaluate(model, train_loader, test_loader)
    activation_scores_try.append((train_accuracies, test_accuracies))
    activation_losses_try.append((train_losses, test_losses))

def save_to_log(filename, model_type, activation_functions, activation_scores, activation_losses):
    with open(filename, 'a') as f:
        f.write(f"Model Type: {model_type}\n")
        for i, activation_function in enumerate(activation_functions):
            f.write(f"Activation Function: {activation_function.__class__.__name__}\n")
            train_scores, test_scores = activation_scores[i]
            train_losses, test_losses = activation_losses[i]
            f.write("Epoch\tTrain Accuracy\tTest Accuracy\tTrain Loss\tTest Loss\n")
            for epoch in range(len(train_scores)):
                f.write(f"{epoch+1}\t{train_scores[epoch]:.4f}\t{test_scores[epoch]:.4f}\t{train_losses[epoch]:.6f}\t{test_losses[epoch]:.6f}\n")
            f.write("\n")

# Save all activations for Try_TULER to log.txt
save_to_log('Try_TULER_log2.txt', 'Try_TULER', activation_functions, activation_scores_try_tuler, activation_losses_try_tuler)

# Save all activations for Try to log.txt
save_to_log('Try_log2.txt', 'Try', activation_functions, activation_scores_try, activation_losses_try)


# Plotting function for all activations in one plot
def plot_all_activations(activation_scores, activation_losses, activation_functions):
    plt.figure(figsize=(12, 6))
    for i, activation_function in enumerate(activation_functions):
        train_scores, test_scores = activation_scores[i]
        train_losses, test_losses = activation_losses[i]
        x = range(1, len(train_scores) + 1)
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(x, train_scores, label=f"Train - {activation_function.__class__.__name__}")
        plt.plot(x, test_scores, label=f"Test - {activation_function.__class__.__name__} - Test")

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(x, train_losses, label=f"Train - {activation_function.__class__.__name__}")
        plt.plot(x, test_losses, label=f"Test - {activation_function.__class__.__name__}")

    plt.subplot(1, 2, 1)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plot all activations for Try_TULER
plot_all_activations(activation_scores_try_tuler, activation_losses_try_tuler, activation_functions)

# Plot all activations for Try
plot_all_activations(activation_scores_try, activation_losses_try, activation_functions)
