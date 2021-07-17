import torch
import torch.nn as nn
import torch.nn.init as init


class BRCLayer(nn.Module):
    """
    Recurrent Neural Network (single layer) using the Bistable Recurrent Cell
    (see arXiv:2006.05252).
    """

    def __init__(self, input_size, hidden_size):
        """
        Arguments
        ---------
        - intput_size: int
            Input size for each element of the sequence
        - hidden_size: int
            Hidden state size
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate
        U_c = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_c = nn.Parameter(U_c)
        self.w_c = nn.Parameter(init.normal_(torch.empty(hidden_size)))
        self.b_c = nn.Parameter(init.normal_(torch.empty(hidden_size)))

        # Reset gate
        U_a = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_a = nn.Parameter(U_a)
        self.w_a = nn.Parameter(init.normal_(torch.empty(hidden_size)))
        self.b_a = nn.Parameter(init.normal_(torch.empty(hidden_size)))

        # Hidden state
        U_h = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_h = nn.Parameter(U_h)
        self.b_h = nn.Parameter(init.normal_(torch.empty(hidden_size)))

    def forward(self, x_seq, h):
        """
        Compute the forward pass for the whole sequence.

        Arguments
        ---------
        - x_seq: tensor of shape (seq_len, batch_size, input_size)
            Input sequence
        - h: tensor of shape (batch_size, hidden_size)
            The eventual initial hidden state at the moment of receiving the
            input.

        Returns
        -------
        - output: tensor of shape (seq_len, batch_size, hidden_size)
            It contains the output of the last layer for all elements of the
            input sequence
        - hn: tensor of shape (batch_size, hidden_size)
            Hidden state at the end of the sequence for all layers of the RNN
        """
        assert h.size(0) == x_seq.size(1)
        assert h.size(1) == self.hidden_size
        assert x_seq.size(2) == self.input_size

        seq_len = x_seq.size(0)
        batch_size = x_seq.size(1)

        y_seq = torch.empty(seq_len, batch_size, self.hidden_size,
                device=x_seq.device)

        for t in range(seq_len):
            x = x_seq[t, :, :]
            c = torch.sigmoid(torch.mm(x, self.U_c.T) + self.w_c * h +
                    self.b_c)
            a = 1. + torch.tanh(torch.mm(x, self.U_a.T) + self.w_a * h +
                    self.b_a)
            h = c * h + (1. - c) * torch.tanh(torch.mm(x, self.U_h.T) + a * h +
                    self.b_h)
            y_seq[t, ...] = h

        return y_seq, h


class nBRCLayer(nn.Module):
    """
    Recurrent Neural Network (single layer) using the Recurrently
    Neuromodulated Bistable Recurrent Cell (see arXiv:2006.05252).
    """

    def __init__(self, input_size, hidden_size):
        """
        Arguments
        ---------
        - intput_size: int
            Input size for each element of the sequence
        - hidden_size: int
            Hidden state size
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate
        U_c = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_c = nn.Parameter(U_c)
        W_c = init.xavier_uniform_(torch.empty(hidden_size, hidden_size))
        self.W_c = nn.Parameter(W_c)
        self.b_c = nn.Parameter(init.normal_(torch.empty(hidden_size)))

        # Reset gate
        U_a = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_a = nn.Parameter(U_a)
        W_a = init.xavier_uniform_(torch.empty(hidden_size, hidden_size))
        self.W_a = nn.Parameter(W_a)
        self.b_a = nn.Parameter(init.normal_(torch.empty(hidden_size)))

        # Hidden state
        U_h = init.xavier_uniform_(torch.empty(hidden_size, input_size))
        self.U_h = nn.Parameter(U_h)
        self.b_h = nn.Parameter(init.normal_(torch.empty(hidden_size)))

    def forward(self, x_seq, h):
        """
        Compute the forward pass for the whole sequence.

        Arguments
        ---------
        - x_seq: tensor of shape (seq_len, batch_size, input_size)
            Input sequence
        - h: tensor of shape (batch_size, hidden_size)
            The eventual initial hidden state at the moment of receiving the
            input.

        Returns
        -------
        - output: tensor of shape (seq_len, batch_size, hidden_size)
            It contains the output of the last layer for all elements of the
            input sequence
        - hn: tensor of shape (batch_size, hidden_size)
            Hidden state at the end of the sequence for all layers of the RNN
        """
        assert h.size(0) == x_seq.size(1)
        assert h.size(1) == self.hidden_size
        assert x_seq.size(2) == self.input_size

        seq_len = x_seq.size(0)
        batch_size = x_seq.size(1)

        y_seq = torch.empty(seq_len, batch_size, self.hidden_size,
                device=x_seq.device)

        for t in range(seq_len):
            x = x_seq[t, :, :]
            c = torch.sigmoid(torch.mm(x, self.U_c.T) +
                    torch.mm(h, self.W_c.T) + self.b_c)
            a = 1. + torch.tanh(torch.mm(x, self.U_a.T) +
                    torch.mm(h, self.W_a.T) + self.b_a)
            h = c * h + (1. - c) * torch.tanh(torch.mm(x, self.U_h.T) +
                    a * h + self.b_h)
            y_seq[t, ...] = h

        return y_seq, h


class BRC(nn.Module):
    """
    Recurrent Neural Network using the (Recurrently Neuromodulated) Bistable
    Recurrent Cell (see arXiv:2006.05252), with several stacked (n)BRC.
    """
    def __init__(self, input_size, hidden_size, num_layers,
            neuromodulated=False, train_h0=False):
        """
        Arguments
        ---------
        - intput_size: int
            Input size for each element of the sequence
        - hidden_size: int
            Hidden state size
        - num_layers: int
            Number of stacked RNNs
        - neuromodulated: bool
            Whether to use neuromodulation (i.e. NBRCLayer instead of BRCLayer)
        - train_h0: bool
            Whether to consider the initial hidden state as a parameter to
            train instead of a fixed zero tensor
        """
        super().__init__()
        self.initial_hidden = torch.zeros(num_layers, hidden_size)
        if train_h0:
            self.initial_hidden = nn.Parameter(self.initial_hidden)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        Layer = nBRCLayer if neuromodulated else BRCLayer
        layers = [Layer(input_size, hidden_size)]
        for _ in range(self.num_layers - 1):
            layers.append(Layer(hidden_size, hidden_size))

        self.layers = nn.ModuleList(layers)

    def forward(self, x_seq, h0=None):
        """
        Compute the forward pass for the whole sequence and along each layers.

        Arguments
        ---------
        - x: tensor of shape (seq_len, batch_size, input_size)
            Input sequence
        - h0: tensor of shape (num_layers, batch_size, hidden_size)
            The eventual initial hidden state at the moment of receiving the
            input.

        Returns
        -------
        - output: tensor of shape (seq_len, batch_size, hidden_size)
            It contains the output of the last layer for all elements of the
            input sequence
        - hn: tensor of shape (num_layers, batch_size, hidden_size)
            Hidden state at the end of the sequence for all layers of the RNN
        """
        batch_size = x_seq.size(1)
        if h0 is None:
            h0 = self.initial_hidden.unsqueeze(1).expand(-1, batch_size, -1)

        hn = torch.empty(self.num_layers, batch_size, self.hidden_size,
                device=x_seq.device)

        for l, (layer, h0l) in enumerate(zip(self.layers, h0)):
            x_seq, hnl = layer(x_seq, h0l)
            hn[l, ...] = hnl

        return x_seq, hn


class nBRC(BRC):
    """
    Recurrent Neural Network using the Recurrently Neuromodulated Bistable
    Recurrent Cell (see arXiv:2006.05252), with several stacked (n)BRC.
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__(input_size, hidden_size, num_layers,
                neuromodulated=True)
