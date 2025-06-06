import torch
from torch import nn

class ACTModule(nn.Module):
    """Adaptive Computation Time (ACT) for the Universal Transformer."""

    def __init__(self, hidden_size: int, max_steps: int = 10, halt_threshold: float = 0.99):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.p = nn.Linear(hidden_size, 1)
        self.max_steps = max_steps
        self.threshold = halt_threshold

    def forward(self, state, halting_prob, remainders, n_updates):
        """Update halting statistics for a single recurrent step."""
        p = self.sigmoid(self.p(state))
        still_running = (halting_prob < 1.0).float()
        new_halted = ((halting_prob + p * still_running) > self.threshold).float() * still_running
        still_running = ((halting_prob + p * still_running) <= self.threshold).float() * still_running

        halting_prob = halting_prob + p * still_running + new_halted * (1 - halting_prob)
        remainders = remainders + new_halted * (1 - remainders)
        n_updates = n_updates + still_running + new_halted
        update_weight = p * still_running + new_halted * (1 - halting_prob)
        return halting_prob, remainders, n_updates, update_weight

class TransformerEncoderLayer(nn.Module):
    """Wrapper around ``nn.TransformerEncoderLayer`` with batch-first support."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model, nhead,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout, batch_first=True)

    def forward(self, x):
        return self.layer(x)

class UTEncoder(nn.Module):
    """Universal Transformer encoder with ACT."""

    def __init__(self, d_model: int, nhead: int, max_steps: int = 5, halt_threshold: float = 0.99):
        super().__init__()
        self.layer = TransformerEncoderLayer(d_model, nhead)
        self.act = ACTModule(d_model, max_steps=max_steps, halt_threshold=halt_threshold)
        self.max_steps = max_steps
        self.threshold = halt_threshold

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        halting_prob = x.new_zeros(batch_size, seq_len, 1)
        remainders = x.new_zeros(batch_size, seq_len, 1)
        n_updates = x.new_zeros(batch_size, seq_len, 1)
        previous_state = x

        for step in range(self.max_steps):
            state = self.layer(previous_state)
            halting_prob, remainders, n_updates, update_weight = self.act(
                state, halting_prob, remainders, n_updates)
            update_weight = update_weight.expand_as(state)
            previous_state = previous_state * (1 - update_weight) + state * update_weight
            if torch.all(halting_prob >= self.threshold):
                break
        return previous_state, (remainders, n_updates)

def example_usage():
    """Demonstrates a tiny training loop on random time-series data."""
    batch_size, seq_len, feature_dim = 8, 15, 32
    model = UTEncoder(d_model=feature_dim, nhead=4, max_steps=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(5):
        data = torch.randn(batch_size, seq_len, feature_dim)
        target = torch.randn(batch_size, seq_len, feature_dim)
        output, _ = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"loss: {loss.item():.4f}")

if __name__ == "__main__":
    example_usage()
