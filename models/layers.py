
import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """
    Custom Dropout from scratch. Rules:
    - NO use of nn.Dropout or F.dropout anywhere
    - Uses inverted dropout: divide kept activations by (1-p)
      so the expected value stays the same at train time
    - At eval time (self.training = False): pass input through unchanged
    Design choice: placed AFTER BatchNorm in FC layers because BN already
    normalizes activations; dropout then randomly silences them to prevent
    co-adaptation of neurons without interfering with BN statistics.
    """

    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1, "Dropout probability p must be in [0, 1)"
        self.p = p

    def forward(self, x):
        # Eval mode: identity — no dropout
        if not self.training:
            return x
        # p=0 means keep everything
        if self.p == 0:
            return x
        # Step 1: random binary mask  (1 = keep, 0 = drop)
        # torch.rand_like gives uniform [0,1); > p gives True with prob (1-p)
        mask = (torch.rand_like(x) > self.p).float()
        # Step 2: inverted dropout scaling so E[output] = E[input]
        return x * mask / (1.0 - self.p)

    def extra_repr(self):
        return f'p={self.p}'
