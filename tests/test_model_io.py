import json

import torch

from kd_pipeline.model_io import load_model, save_checkpoint
from kd_pipeline.models import LightNN


def test_save_and_load_round_trip_preserves_weights(tmp_path):
    model = LightNN(num_classes=10)
    checkpoint_path = tmp_path / "student.pt"

    save_checkpoint(model, checkpoint_path, arch="LightNN", metadata={"test_accuracy": 70.5})

    loaded = load_model("LightNN", checkpoint_path, device="cpu")

    for p_orig, p_loaded in zip(model.parameters(), loaded.parameters()):
        assert torch.allclose(p_orig, p_loaded)


def test_save_checkpoint_writes_metadata_sidecar(tmp_path):
    model = LightNN(num_classes=10)
    checkpoint_path = tmp_path / "student.pt"

    save_checkpoint(model, checkpoint_path, arch="LightNN", metadata={"test_accuracy": 70.5})

    sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    assert sidecar_path.is_file()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["arch"] == "LightNN"
    assert payload["test_accuracy"] == 70.5


def test_checkpoint_is_state_dict_not_whole_module(tmp_path):
    """Guards against regressing to the notebook's pickle.dump(model) pattern."""
    model = LightNN(num_classes=10)
    checkpoint_path = tmp_path / "student.pt"
    save_checkpoint(model, checkpoint_path, arch="LightNN")

    loaded_object = torch.load(checkpoint_path, map_location="cpu")
    assert isinstance(loaded_object, dict)
    assert not isinstance(loaded_object, torch.nn.Module)
