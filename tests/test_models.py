import torch

from src.models import DeepCNN, SimpleCNN, TransferModel


def test_simple_cnn_output_shape():
    output = SimpleCNN()(torch.randn(2, 3, 128, 128))
    assert output.shape == (2, 28)


def test_deep_cnn_output_shape():
    output = DeepCNN()(torch.randn(2, 3, 128, 128))
    assert output.shape == (2, 28)


def test_transfer_model_output_shape_without_download():
    model = TransferModel(pretrained=False)
    output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 28)
