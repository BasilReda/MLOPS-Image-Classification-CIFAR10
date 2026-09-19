import torch

from kd_pipeline.models import DeepNN, LightNN, build_model


def test_deepnn_forward_pass_shape():
    model = DeepNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_lightnn_forward_pass_shape():
    model = LightNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    assert out.shape == (4, 10)


def test_deepnn_param_count_regression_guard():
    model = DeepNN(num_classes=10)
    total = sum(p.numel() for p in model.parameters())
    assert total == 1_186_986


def test_lightnn_param_count_regression_guard():
    model = LightNN(num_classes=10)
    total = sum(p.numel() for p in model.parameters())
    assert total == 267_738


def test_build_model_factory():
    assert isinstance(build_model("DeepNN"), DeepNN)
    assert isinstance(build_model("LightNN"), LightNN)


def test_build_model_unknown_arch_raises():
    try:
        build_model("NoSuchArch")
        assert False, "expected ValueError"
    except ValueError:
        pass
