import torch

from src.models import SimpleNN
from src.utils import load_model, save_model, set_seed


def test_seed_is_reproducible():
    set_seed(123)
    first = torch.rand(4)
    set_seed(123)
    second = torch.rand(4)
    assert torch.equal(first, second)


def test_model_checkpoint_roundtrip(tmp_path):
    model = SimpleNN()
    checkpoint = tmp_path / "model.pth"
    save_model(model, str(checkpoint))
    loaded = load_model(SimpleNN, str(checkpoint))
    for expected, actual in zip(model.parameters(), loaded.parameters()):
        assert torch.equal(expected, actual)
