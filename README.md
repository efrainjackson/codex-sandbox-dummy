# codex-sandbox-dummy

This repository contains minimal examples. A Universal Transformer with Adaptive Computation Time (ACT) is implemented in `models/universal_transformer_act.py`.

## Requirements

- [PyTorch](https://pytorch.org) (tested with 2.x)

Install PyTorch following the instructions for your platform.

## Running the Example

The encoder can be exercised with a small training loop using randomly generated time-series data. Simply run:

```bash
python models/universal_transformer_act.py
```

The script will print the loss for a few iterations demonstrating how to feed data shaped `[batch, seq_len, feature_dim]` into the encoder.
