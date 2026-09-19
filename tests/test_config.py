import pytest

from kd_pipeline.config import load_params


def test_load_params_reads_repo_params_yaml():
    params = load_params()
    assert "seed" in params
    assert "data" in params
    assert "teacher" in params
    assert "distill" in params
    assert params["distill"]["temperature"] == 2


def test_load_params_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_params(tmp_path / "does_not_exist.yaml")


def test_load_params_rejects_non_mapping(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_params(bad)
