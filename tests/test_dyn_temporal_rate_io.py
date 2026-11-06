"""Pins for the temporal-rate clock-dataset interchange formats."""
from __future__ import annotations

import json

import numpy as np
import pytest

from apedp_dyn.epistemic.temporal_rate_io import (
    adjudicate_preregistered_temporal_rate,
    dataset_sha256,
    load_clock_dataset,
    load_preregistration,
    preregister_datasets,
    save_clock_dataset,
    save_preregistration,
    verify_preregistration,
)
from apedp_dyn.epistemic.temporal_rate_modulation import (
    TemporalRateProtocolError,
    TemporalRateVerdict,
    adjudicate_temporal_rate_modulation,
    simulate_clock_dataset,
)


def _dataset(seed: int = 1):
    return simulate_clock_dataset(
        dataset_id="run-1",
        drive_frequency_hz=1.0,
        duration_s=32.0,
        sample_rate_hz=16.0,
        seed=seed,
    )


def _pair(seed: int = 1):
    calibration = simulate_clock_dataset(
        dataset_id="cal",
        drive_frequency_hz=0.83,
        duration_s=32.0,
        sample_rate_hz=16.0,
        seed=seed,
    )
    confirmation = simulate_clock_dataset(
        dataset_id="conf",
        drive_frequency_hz=1.0,
        duration_s=32.0,
        sample_rate_hz=16.0,
        seed=seed + 1,
    )
    return calibration, confirmation


def _preregistration(tmp_path, calibration, confirmation):
    preregistration = preregister_datasets(
        calibration,
        confirmation,
        notes="frozen before adjudication",
        created_utc="2026-10-28T00:00:00+00:00",
    )
    path = tmp_path / "preregistration.json"
    save_preregistration(preregistration, path)
    return preregistration, path


def test_csv_round_trip_preserves_data_and_fingerprint(tmp_path):
    dataset = _dataset()
    path = tmp_path / "run.csv"
    save_clock_dataset(dataset, path)

    loaded = load_clock_dataset(path)

    assert loaded.dataset_id == dataset.dataset_id
    assert loaded.drive_frequency_hz == dataset.drive_frequency_hz
    np.testing.assert_array_equal(loaded.time_s, dataset.time_s)
    np.testing.assert_array_equal(loaded.drive, dataset.drive)
    for left, right in zip(loaded.channels, dataset.channels, strict=True):
        assert left.clock_id == right.clock_id
        assert left.composition == right.composition
        assert (
            left.systematic_susceptibility_sigma
            == right.systematic_susceptibility_sigma
        )
        np.testing.assert_array_equal(
            left.fractional_frequency, right.fractional_frequency
        )
    assert dataset_sha256(loaded) == dataset_sha256(dataset)


def test_npz_round_trip_preserves_data_and_fingerprint(tmp_path):
    dataset = _dataset(seed=2)
    path = tmp_path / "run.npz"
    save_clock_dataset(dataset, path)

    loaded = load_clock_dataset(path)

    assert loaded.dataset_id == dataset.dataset_id
    np.testing.assert_array_equal(loaded.time_s, dataset.time_s)
    np.testing.assert_array_equal(loaded.drive, dataset.drive)
    for left, right in zip(loaded.channels, dataset.channels, strict=True):
        np.testing.assert_array_equal(
            left.fractional_frequency, right.fractional_frequency
        )
    assert dataset_sha256(loaded) == dataset_sha256(dataset)


def test_csv_and_npz_load_to_the_same_fingerprint(tmp_path):
    dataset = _dataset(seed=3)
    csv_path = tmp_path / "run.csv"
    npz_path = tmp_path / "run.npz"
    save_clock_dataset(dataset, csv_path)
    save_clock_dataset(dataset, npz_path)

    assert dataset_sha256(load_clock_dataset(csv_path)) == dataset_sha256(
        load_clock_dataset(npz_path)
    )


def test_fingerprint_changes_with_data():
    assert dataset_sha256(_dataset(seed=4)) != dataset_sha256(_dataset(seed=5))


def test_csv_sidecar_is_required(tmp_path):
    path = tmp_path / "missing.csv"
    path.write_text(
        "time_s,drive,clock_0_fractional_frequency\n0,1,0\n1,1,0\n",
        encoding="utf-8",
    )
    with pytest.raises(TemporalRateProtocolError, match="metadata"):
        load_clock_dataset(path)


def test_metadata_rejects_missing_and_unknown_keys(tmp_path):
    dataset = _dataset()
    path = tmp_path / "run.csv"
    save_clock_dataset(dataset, path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    metadata.pop("drive_frequency_hz")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(TemporalRateProtocolError, match="missing"):
        load_clock_dataset(path)

    metadata["drive_frequency_hz"] = 1.0
    metadata["unexpected"] = 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(TemporalRateProtocolError, match="unknown"):
        load_clock_dataset(path)


def test_metadata_rejects_duplicate_clock_ids(tmp_path):
    dataset = _dataset()
    path = tmp_path / "run.csv"
    save_clock_dataset(dataset, path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["clocks"][1]["clock_id"] = metadata["clocks"][0]["clock_id"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(TemporalRateProtocolError, match="duplicate clock_id"):
        load_clock_dataset(path)


def test_csv_rejects_empty_and_nonfinite_cells(tmp_path):
    dataset = _dataset()
    path = tmp_path / "run.csv"
    save_clock_dataset(dataset, path)
    lines = path.read_text(encoding="utf-8").splitlines()

    lines[1] = lines[1].rsplit(",", 1)[0] + ","
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(TemporalRateProtocolError, match="is empty"):
        load_clock_dataset(path)

    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].rsplit(",", 1)[0] + ",nan"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(TemporalRateProtocolError, match="is not finite"):
        load_clock_dataset(path)


def test_npz_rejects_unknown_keys(tmp_path):
    valid_path = tmp_path / "valid.npz"
    save_clock_dataset(_dataset(), valid_path)
    with np.load(valid_path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["extra"] = np.array([1.0])

    path = tmp_path / "unknown.npz"
    np.savez_compressed(path, **payload)
    with pytest.raises(TemporalRateProtocolError, match="unknown"):
        load_clock_dataset(path)


def test_npz_rejects_object_arrays(tmp_path):
    path = tmp_path / "object.npz"
    np.savez_compressed(
        path,
        time_s=np.array([0.0, 1.0]),
        drive=np.array([1.0, 1.0]),
        drive_frequency_hz=np.float64(1.0),
        dataset_id=np.array("object"),
        clock_ids=np.array(["clock_0"]),
        compositions=np.array(["Sr"]),
        fractional_frequency=np.array([[0.0, 0.0]], dtype=object),
        systematic_susceptibility_sigma=np.array([0.0]),
    )
    with pytest.raises(TemporalRateProtocolError, match="pickle"):
        load_clock_dataset(path)


def test_npz_metadata_path_is_rejected(tmp_path):
    path = tmp_path / "run.npz"
    save_clock_dataset(_dataset(), path)
    with pytest.raises(TemporalRateProtocolError, match="metadata_path"):
        load_clock_dataset(path, metadata_path=tmp_path / "run.json")


def test_single_clock_round_trips_through_npz(tmp_path):
    dataset = simulate_clock_dataset(
        dataset_id="single",
        drive_frequency_hz=1.0,
        duration_s=32.0,
        sample_rate_hz=16.0,
        compositions=("Sr",),
        susceptibilities=(1e-15 + 0j,),
        seed=6,
    )
    path = tmp_path / "single.npz"
    save_clock_dataset(dataset, path)

    loaded = load_clock_dataset(path)

    assert len(loaded.channels) == 1
    assert loaded.channels[0].clock_id == "clock_0"
    assert dataset_sha256(loaded) == dataset_sha256(dataset)


def test_unsupported_format_is_refused(tmp_path):
    with pytest.raises(TemporalRateProtocolError, match="unsupported"):
        load_clock_dataset(tmp_path / "run.txt")


def test_preregistration_round_trip_and_verification(tmp_path):
    calibration, confirmation = _pair()
    preregistration, path = _preregistration(
        tmp_path, calibration, confirmation
    )

    loaded = load_preregistration(path)

    assert loaded == preregistration
    verify_preregistration(loaded, calibration, confirmation)


def test_preregistration_rejects_swapped_confirmation(tmp_path):
    calibration, confirmation = _pair()
    preregistration, _ = _preregistration(
        tmp_path, calibration, confirmation
    )
    swapped = simulate_clock_dataset(
        dataset_id="conf",
        drive_frequency_hz=1.0,
        duration_s=32.0,
        sample_rate_hz=16.0,
        seed=3,
    )

    with pytest.raises(
        TemporalRateProtocolError, match="confirmation_sha256"
    ):
        verify_preregistration(preregistration, calibration, swapped)


def test_preregistration_rejects_tampered_calibration(tmp_path):
    calibration, confirmation = _pair()
    preregistration, _ = _preregistration(
        tmp_path, calibration, confirmation
    )
    tampered = simulate_clock_dataset(
        dataset_id="cal",
        drive_frequency_hz=0.83,
        duration_s=32.0,
        sample_rate_hz=16.0,
        seed=4,
    )

    with pytest.raises(
        TemporalRateProtocolError, match="calibration_sha256"
    ):
        verify_preregistration(preregistration, tampered, confirmation)


def test_preregistration_rejects_changed_metadata(tmp_path):
    calibration, confirmation = _pair()
    _, path = _preregistration(tmp_path, calibration, confirmation)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["confirmation_drive_frequency_hz"] = 2.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_preregistration(path)
    with pytest.raises(
        TemporalRateProtocolError, match="confirmation_drive_frequency_hz"
    ):
        verify_preregistration(loaded, calibration, confirmation)


def test_preregistration_rejects_unknown_keys(tmp_path):
    calibration, confirmation = _pair()
    _, path = _preregistration(tmp_path, calibration, confirmation)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["extra"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TemporalRateProtocolError, match="unknown"):
        load_preregistration(path)


def test_preregistration_rejects_unsupported_protocol_version(tmp_path):
    calibration, confirmation = _pair()
    _, path = _preregistration(tmp_path, calibration, confirmation)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol_version"] = "temporal-rate-v1"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_preregistration(path)
    with pytest.raises(TemporalRateProtocolError, match="protocol_version"):
        verify_preregistration(loaded, calibration, confirmation)


def test_preregistration_requires_distinct_dataset_ids():
    calibration, _ = _pair()
    same_id = simulate_clock_dataset(
        dataset_id="cal",
        drive_frequency_hz=1.0,
        duration_s=32.0,
        sample_rate_hz=16.0,
        seed=5,
    )
    with pytest.raises(TemporalRateProtocolError, match="dataset_id"):
        preregister_datasets(calibration, same_id)


def test_preregistered_adjudication_returns_verdict(tmp_path):
    calibration, _ = _pair()
    confirmation = simulate_clock_dataset(
        dataset_id="conf",
        drive_frequency_hz=1.0,
        duration_s=32.0,
        sample_rate_hz=16.0,
        anomaly_amplitude=1e-15,
        anomaly_phase_rad=1.1,
        seed=6,
    )
    preregistration, _ = _preregistration(
        tmp_path, calibration, confirmation
    )

    result = adjudicate_preregistered_temporal_rate(
        preregistration,
        calibration,
        confirmation,
    )

    assert result.verdict is TemporalRateVerdict.COMMON_MODE_ANOMALY_CANDIDATE


def test_preregistration_freezes_analysis_parameters(tmp_path):
    calibration, confirmation = _pair()
    preregistration = preregister_datasets(
        calibration,
        confirmation,
        harmonics=2,
        n_surrogates=64,
        seed=7,
        notes="frozen analysis configuration",
        created_utc="2026-10-28T00:00:00+00:00",
    )
    path = tmp_path / "frozen.json"
    save_preregistration(preregistration, path)
    loaded = load_preregistration(path)

    assert loaded.harmonics == 2
    assert loaded.n_surrogates == 64
    assert loaded.seed == 7
    assert loaded.familywise_alpha == pytest.approx(0.01)
    assert loaded.heterogeneity_alpha == pytest.approx(0.01)
    assert loaded.min_fractional_amplitude == pytest.approx(1e-18)
    assert len(loaded.canonical_spec_sha256) == 64

    preregistered_result = adjudicate_preregistered_temporal_rate(
        loaded, calibration, confirmation
    )
    explicit_result = adjudicate_temporal_rate_modulation(
        calibration,
        confirmation,
        harmonics=2,
        n_surrogates=64,
        seed=7,
    )
    assert preregistered_result.p_value == explicit_result.p_value
    assert preregistered_result.harmonics == 2
    assert preregistered_result.n_surrogates == 64


def test_preregistered_adjudication_has_no_parameter_overrides():
    import inspect

    signature = inspect.signature(adjudicate_preregistered_temporal_rate)
    assert set(signature.parameters) == {
        "preregistration",
        "calibration",
        "confirmation",
    }


def test_preregistration_rejects_tampered_canonical_spec_hash(tmp_path):
    calibration, confirmation = _pair()
    _, path = _preregistration(tmp_path, calibration, confirmation)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["canonical_spec_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_preregistration(path)
    with pytest.raises(
        TemporalRateProtocolError, match="canonical_spec_sha256"
    ):
        verify_preregistration(loaded, calibration, confirmation)


def test_preregistration_rejects_tampered_canonical_alpha(tmp_path):
    calibration, confirmation = _pair()
    _, path = _preregistration(tmp_path, calibration, confirmation)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["familywise_alpha"] = 0.05
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_preregistration(path)
    with pytest.raises(TemporalRateProtocolError, match="familywise_alpha"):
        verify_preregistration(loaded, calibration, confirmation)


def test_preregistration_requires_distinct_compositions():
    calibration = simulate_clock_dataset(
        dataset_id="same-composition-cal",
        drive_frequency_hz=0.83,
        duration_s=32.0,
        sample_rate_hz=16.0,
        compositions=("Sr", "Sr"),
        susceptibilities=(1e-15 + 0j, 2e-15 + 0j),
        seed=1,
    )
    confirmation = simulate_clock_dataset(
        dataset_id="same-composition-conf",
        drive_frequency_hz=1.0,
        duration_s=32.0,
        sample_rate_hz=16.0,
        compositions=("Sr", "Sr"),
        susceptibilities=(1e-15 + 0j, 2e-15 + 0j),
        seed=2,
    )
    with pytest.raises(
        TemporalRateProtocolError, match="physically distinct"
    ):
        preregister_datasets(calibration, confirmation)
