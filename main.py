import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from brc import BRC, nBRC


# Problem hyperparameters
SEQ_LEN = 32
NUM_SEQ = 65536

# Architecture hyperparameters
HIDDEN_SIZE = 32
NUM_LAYERS = 1

# Training hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 256


class RecurrentNeuralNetwork(nn.Module):
    """
    Wrapper around the recurrent cell with a linear layer before the output.
    """
    def __init__(self, cell, hidden_size, output_size):
        super().__init__()
        self.cell = cell
        self.linear = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x, _ = self.cell(x)
        return self.linear(x)


if __name__ == '__main__':
    """
    The RNN is trained to copy the first input.
    """

    # CPU is usually faster for RNNs
    device = torch.device('cpu')

    # Generate random input sequences
    x_seq = torch.randn(SEQ_LEN, NUM_SEQ, 1, device=device)

    # Initialise architecture and optimiser
    brc = BRC(1, HIDDEN_SIZE, NUM_LAYERS)
    rnn = RecurrentNeuralNetwork(brc, HIDDEN_SIZE, 1)
    rnn.to(device)
    optimiser = optim.Adam(rnn.parameters(), lr=1e-3)

    # Log results
    losses = []
    with open('results.csv', 'w') as f:
        f.write('epoch,loss\n')

    # Training loop
    for e in range(NUM_EPOCHS):
        try:
            # Shuffle samples
            permutation = torch.randperm(NUM_SEQ)

            # Track mean loss per epoch
            mean_loss = 0.
            for i in range(int(NUM_SEQ / BATCH_SIZE)):

                # Generate minibatch
                indices = permutation[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                batch_input = x_seq[:, indices, :]
                batch_output = batch_input[0, :, :]

                # Compute prediction and loss on last time step
                batch_pred = rnn(batch_input)
                loss = F.mse_loss(batch_pred[-1, ...], batch_output)

                # Optimise on the minibatch
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # Track mean loss per epoch
                mean_loss += loss.item()

            # Divide total loss by number of minibatches
            mean_loss /= (NUM_SEQ / BATCH_SIZE)

            # Log results
            losses.append(mean_loss)
            with open('results.csv', 'a') as f:
                f.write('{},{}\n'.format(e + 1, mean_loss))
            print("Epoch {}. MSE Loss: {:.4f}".format(e + 1, mean_loss))

        except KeyboardInterrupt:

            print('\nTraining interrupted, display results...')
            break

    # Display and save results
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Mean loss')
    plt.tight_layout()
    plt.savefig('results.pdf', transparent=True)
    plt.show()
