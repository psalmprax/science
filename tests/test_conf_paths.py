from __future__ import annotations

from apedp_dyn.conf.paths import artifact_dir


def test_artifact_dir_creates_default_and_override_paths(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    override = tmp_path / "custom" / "artifacts"
    monkeypatch.setenv("APEDP_REPO_ROOT", str(repo))
    monkeypatch.delenv("APEDP_ARTIFACT_DIR", raising=False)

    assert artifact_dir() == repo / "artifacts"
    assert (repo / "artifacts").is_dir()

    monkeypatch.setenv("APEDP_ARTIFACT_DIR", str(override))
    assert artifact_dir() == override
    assert override.is_dir()
