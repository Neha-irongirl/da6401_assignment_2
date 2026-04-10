
import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """
    Custom Dropout built from scratch — nn.Dropout is NOT used here.

    How it works:
      - During TRAINING: randomly sets some values to 0 with probability p
        Then scales remaining values by 1/(1-p) so the average stays the same
        This is called 'inverted dropout'
      - During EVALUATION (testing): does nothing — passes input unchanged
        Because at test time we want all neurons active

    Example:
      p=0.5 means 50% of neurons are randomly turned off each forward pass
    """

    def __init__(self, p=0.5):
        super().__init__()
        assert 0 <= p < 1, "Dropout probability p must be between 0 and 1"
        self.p = p

    def forward(self, x):
        # If we are in eval mode OR p=0, do nothing
        if not self.training or self.p == 0:
            return x

        # Create a random mask — same shape as input
        # Each value is True (keep) with probability (1-p)
        # False (drop) with probability p
        mask = (torch.rand_like(x) > self.p).float()

        # Inverted scaling: divide by (1-p)
        # This keeps the expected sum of activations the same
        # Example: if p=0.5, we scale by 2 to compensate for dropped neurons
        return x * mask / (1.0 - self.p)

    def extra_repr(self):
        # This just makes print(model) show the p value nicely
        return f'p={self.p}'
