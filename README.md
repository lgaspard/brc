# PyTorch Implementation of (n)BRC

PyTorch implementation of the bistable recurrent cell (BRC) and recurrently
neuromodulated bistable recurrent cell (nBRC).

The available classes, `BRCLayer`, `nBRCLayer`, `BRC` and `nBRC`, are
documented in [brc.py](brc.py).

## Download

```
git clone https://github.com/lgaspard/brc
cd brc/
```

## Example usage

See [main.py](main.py) for a *copy-first-input* benchmark with the BRC cell.
```
python3 main.py
```

## Notes

The implementation is similar to that of `torch.nn.GRU`, such that the output
of the RNN is its hidden state.  A small wrapper is proposed in
[main.py](main.py) to add a linear layer on top of the recurrent cell.

Also note that the parameter `train_h0` allows to make the initial hidden state
a trainable parameter of the recurrent cell.
