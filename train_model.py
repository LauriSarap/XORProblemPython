import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time


def train(SEED, ACTIVATION_FUNCTION, PARAMETER_SAVING_FREQUENCY=1000):

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    #print(device)

    input_size = 2
    hidden_size = 2
    output_size = 1
    num_epochs = 50000
    learning_rate = 0.1
    parameter_saving_frequency = PARAMETER_SAVING_FREQUENCY
    loss_print_frequency = 1000

    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    X = X.to(device)
    Y = Y.to(device)

    # Neural Network Model
    class NeuralNet(nn.Module):
        def __init__(self):
            super(NeuralNet, self).__init__()
            self.layer1 = nn.Linear(input_size, hidden_size)
            self.layer2 = nn.Linear(hidden_size, output_size)

            # Activation functions
            if ACTIVATION_FUNCTION == 'tanh':
                self.activation = nn.Tanh()
            elif ACTIVATION_FUNCTION == 'sigmoid':
                self.activation = nn.Sigmoid()
            elif ACTIVATION_FUNCTION == 'relu':
                self.activation = nn.ReLU()
            elif ACTIVATION_FUNCTION == 'leaky_relu':
                self.activation = nn.LeakyReLU()
            elif ACTIVATION_FUNCTION == 'swish':
                self.activation = nn.SiLU()

            # Xavier initialization
            nn.init.xavier_normal_(self.layer1.weight, 1)
            nn.init.xavier_normal_(self.layer2.weight, 1)

            # Initialize biases to zeros
            self.layer1.bias.data.fill_(0.0)
            self.layer2.bias.data.fill_(0.0)

        def forward(self, x):
            h = self.activation(self.layer1(x))
            y = self.activation(self.layer2(h))
            return y

    model = NeuralNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Model parameters over epochs
    l1_weights_epoch = []
    l1_biases_epoch = []
    l2_weights_epoch = []
    l2_biases_epoch = []

    l1_weight_gradients_epoch = []
    l1_bias_gradients_epoch = []
    l2_weight_gradients_epoch = []
    l2_bias_gradients_epoch = []

    losses = []
    predicted_outputs = []

    # Record initial weights, biases, and gradients
    l1_weights_epoch.append(model.layer1.weight.data.clone().cpu().numpy())
    l1_biases_epoch.append(model.layer1.bias.data.clone().cpu().numpy())
    l2_weights_epoch.append(model.layer2.weight.data.clone().cpu().numpy())
    l2_biases_epoch.append(model.layer2.bias.data.clone().cpu().numpy())

    l1_weight_gradients_epoch.append(np.zeros_like(model.layer1.weight.data.clone().cpu().numpy()))
    l1_bias_gradients_epoch.append(np.zeros_like(model.layer1.bias.data.clone().cpu().numpy()))
    l2_weight_gradients_epoch.append(np.zeros_like(model.layer2.weight.data.clone().cpu().numpy()))
    l2_bias_gradients_epoch.append(np.zeros_like(model.layer2.bias.data.clone().cpu().numpy()))

    losses.append(criterion(model(X), Y).item())
    predicted_outputs.append(model(X).detach().clone().cpu().numpy())

    start_time = time.time()

    for epoch in range(num_epochs):
        # Forward propagation
        outputs = model(X)
        loss = criterion(outputs, Y)

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if (epoch + 1) % loss_print_frequency == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.12f}, Time: {time.time() - start_time:.4f}s')

        if (epoch + 1) % parameter_saving_frequency == 0:
            l1_weights_epoch.append(model.layer1.weight.data.clone().cpu().numpy())
            l1_biases_epoch.append(model.layer1.bias.data.clone().cpu().numpy())
            l2_weights_epoch.append(model.layer2.weight.data.clone().cpu().numpy())
            l2_biases_epoch.append(model.layer2.bias.data.clone().cpu().numpy())

            l1_weight_gradients_epoch.append(model.layer1.weight.grad.clone().cpu().numpy())
            l1_bias_gradients_epoch.append(model.layer1.bias.grad.clone().cpu().numpy())
            l2_weight_gradients_epoch.append(model.layer2.weight.grad.clone().cpu().numpy())
            l2_bias_gradients_epoch.append(model.layer2.bias.grad.clone().cpu().numpy())

            losses.append(loss.item())
            predicted_outputs.append(outputs.detach().clone().cpu().numpy())

    end_time = time.time()
    print(f"Overall training time: {end_time - start_time:.2f}s")

    # Return the collected data
    data = {
        'l1_weights_epoch': l1_weights_epoch,
        'l1_biases_epoch': l1_biases_epoch,
        'l2_weights_epoch': l2_weights_epoch,
        'l2_biases_epoch': l2_biases_epoch,
        'l1_weight_gradients_epoch': l1_weight_gradients_epoch,
        'l1_bias_gradients_epoch': l1_bias_gradients_epoch,
        'l2_weight_gradients_epoch': l2_weight_gradients_epoch,
        'l2_bias_gradients_epoch': l2_bias_gradients_epoch,
        'losses': losses,
        'predicted_outputs': predicted_outputs,
    }

    return data
