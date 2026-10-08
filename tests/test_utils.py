import torch

from src.models import SimpleCNN
import pytest

from src.utils import get_device, load_model, save_model, set_seed


def test_seed_is_reproducible():
    set_seed(123)
    first = torch.rand(4)
    set_seed(123)
    second = torch.rand(4)
    assert torch.equal(first, second)


def test_get_device_cpu():
    assert get_device("cpu") == torch.device("cpu")


def test_get_device_rejects_unknown_device():
    with pytest.raises(ValueError, match="device"):
        get_device("tpu")


def test_model_checkpoint_roundtrip(tmp_path):
    model = SimpleCNN()
    checkpoint = tmp_path / "model.pth"
    save_model(model, str(checkpoint))
    loaded = load_model(SimpleCNN, str(checkpoint))
    for expected, actual in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(expected, actual)
