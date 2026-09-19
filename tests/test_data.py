import torch
from PIL import Image
from torch.utils.data import TensorDataset

from kd_pipeline.data import denormalize, get_transforms, make_splits


def test_transform_produces_correct_shape_and_normalization():
    tfm = get_transforms(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    tensor = tfm(img)
    assert tensor.shape == (3, 32, 32)
    # ToTensor -> [0,1], then Normalize with mean/std=0.5 maps [0,1] -> [-1,1]
    assert tensor.max().item() <= 1.0 + 1e-6
    assert tensor.min().item() >= -1.0 - 1e-6


def test_denormalize_inverts_normalize():
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    original = torch.rand(3, 4, 4)
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    normalized = (original - mean_t) / std_t
    recovered = denormalize(normalized, mean=mean, std=std)
    assert torch.allclose(recovered, original, atol=1e-5)


def test_make_splits_disjoint_and_correctly_sized():
    n = 100
    images = torch.randn(n, 3, 4, 4)
    labels = torch.randint(0, 10, (n,))
    dataset = TensorDataset(images, labels)

    train_subset, val_subset = make_splits(dataset, val_fraction=0.2, seed=42)

    assert len(val_subset) == 20
    assert len(train_subset) == 80
    assert set(train_subset.indices).isdisjoint(set(val_subset.indices))


def test_make_splits_deterministic_under_fixed_seed():
    n = 50
    dataset = TensorDataset(torch.randn(n, 3, 4, 4), torch.randint(0, 10, (n,)))

    train_a, val_a = make_splits(dataset, val_fraction=0.1, seed=7)
    train_b, val_b = make_splits(dataset, val_fraction=0.1, seed=7)

    assert list(train_a.indices) == list(train_b.indices)
    assert list(val_a.indices) == list(val_b.indices)
