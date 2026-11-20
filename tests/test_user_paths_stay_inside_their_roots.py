"""User-supplied names never steer a file path outside the directory meant for it.

Found by CodeQL (py/path-injection) on PR #62; each case is the input that
would have escaped.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")

from apedp.api.routes.pipeline import _validate_safe_pipeline_path
from apedp_dyn.sandbox.volume_sandbox import load_equity_volume


def test_a_path_under_the_temp_root_is_admitted_in_normal_form(tmp_path):
    target = tmp_path / "data.csv"
    assert _validate_safe_pipeline_path(str(target), "csv_path") == os.path.realpath(target)


@pytest.mark.parametrize("raw", ["/etc/passwd", "../outside.csv", "", "a\x00b"])
def test_paths_outside_the_roots_are_refused(raw):
    with pytest.raises(fastapi.HTTPException) as err:
        _validate_safe_pipeline_path(raw, "csv_path")
    assert err.value.status_code == 400


def test_a_symlink_out_of_the_temp_root_is_refused():
    with tempfile.TemporaryDirectory() as d:
        link = Path(d) / "escape"
        try:
            link.symlink_to("/etc")
        except OSError:
            pytest.skip("symlinks unavailable")
        with pytest.raises(fastapi.HTTPException):
            _validate_safe_pipeline_path(str(link / "passwd"), "csv_path")


def test_a_prefix_lookalike_root_is_not_a_root(tmp_path):
    sibling = Path(os.path.realpath(tempfile.gettempdir()) + "-lookalike") / "x.csv"
    with pytest.raises(fastapi.HTTPException):
        _validate_safe_pipeline_path(str(sibling), "csv_path")


@pytest.mark.parametrize("asset", ["../../etc/passwd", "SPX/../../x", "BTC"])
def test_the_equity_loader_reads_only_its_two_sidecars(asset):
    with pytest.raises(ValueError, match="no equity volume sidecar"):
        load_equity_volume(asset)
