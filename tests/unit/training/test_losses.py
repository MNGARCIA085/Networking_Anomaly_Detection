import pytest
import torch
import torch.nn as nn

from network_anomaly_detection.training.nn.losses import create_loss


def test_create_mse_loss():
    loss = create_loss({"name": "mse"})

    assert isinstance(loss, nn.MSELoss)
