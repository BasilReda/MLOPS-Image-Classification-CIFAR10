import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from kd_pipeline.model_io import save_checkpoint
from kd_pipeline.models import LightNN


@pytest.fixture()
def client(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "student_state_dict.pt"
    save_checkpoint(LightNN(num_classes=10), checkpoint_path, arch="LightNN")
    monkeypatch.setenv("KD_CHECKPOINT_PATH", str(checkpoint_path))

    from api.main import app

    with TestClient(app) as c:
        yield c


def _dummy_image_bytes() -> bytes:
    img = Image.new("RGB", (32, 32), color=(0, 128, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_endpoint_returns_well_formed_response(client):
    resp = client.post(
        "/predict", files={"file": ("test.png", _dummy_image_bytes(), "image/png")}
    )
    assert resp.status_code == 200
    body = resp.json()

    known_classes = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    }
    assert body["class_name"] in known_classes
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top3"]) == 3


def test_predict_endpoint_rejects_non_image(client):
    resp = client.post(
        "/predict", files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    assert resp.status_code == 400
