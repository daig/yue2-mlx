"""A checkpoint identity alone cannot bind a numerical calibration to a request."""
import json
from pathlib import Path

import numpy as np
import pytest


def test_calibration_rejects_different_teacher_forced_tokens(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from limits import derive

    reference, anchor = tmp_path / "reference", tmp_path / "anchor"
    reference.mkdir()
    anchor.mkdir()
    model = {"checkpoint": "same-checkpoint"}
    (reference / "ar.json").write_text(json.dumps({"model": model}))
    (anchor / "invocation.json").write_text(json.dumps({"weights": model}))
    ids = np.array([10, 11, 12, 13, 14], dtype=np.int32)
    np.savez(reference / "ar.npz", ids_5=ids, logits_5=np.zeros((1, 2), dtype=np.float32))
    np.savez(anchor / "ar.npz", ids_5=ids, logits_5=np.ones((1, 2), dtype=np.float32))
    calibrated = derive("ar", reference, anchor)
    assert calibrated["ar"]["tensors"]["logits_5"] == {"rms_error": 2, "max_abs": 2}

    changed = ids.copy()
    changed[-1] += 1
    np.savez(anchor / "ar.npz", ids_5=changed, logits_5=np.ones((1, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        derive("ar", reference, anchor)


def test_nar_calibration_rejects_changed_fixed_state_reference(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from limits import derive
    from yue2.storage import sha256_file

    reference, anchor = tmp_path / "reference", tmp_path / "anchor"
    reference.mkdir()
    anchor.mkdir()
    model = {"checkpoint": "same-checkpoint"}
    (reference / "nar.json").write_text(json.dumps({"model": model}))
    inputs = {
        "prefix": np.array([10, 11], dtype=np.int32),
        "codec": np.array([12], dtype=np.int32),
        "noise": np.zeros((1, 64), dtype=np.float32),
    }
    state = np.ones((1, 64), dtype=np.float32)
    np.savez(reference / "nar.npz", **inputs, state_15=state, velocity_15=state, latents=state)
    (anchor / "invocation.json").write_text(json.dumps({
        "weights": model,
        "reference_sha256": sha256_file(reference / "nar.npz"),
    }))
    np.savez(anchor / "nar.npz", **inputs, fixed_velocity_15=state, latents=state)
    calibrated = derive("nar", reference, anchor)
    assert calibrated["nar"]["tensors"]["fixed_velocity_15"]["max_abs"] == 0

    # A new BF16 capture can have identical tokens/noise but different ODE states.
    # Its velocities must not be paired with an anchor evaluated on the old states.
    np.savez(
        reference / "nar.npz", **inputs,
        state_15=state * 2, velocity_15=state * 2, latents=state,
    )
    with pytest.raises(ValueError):
        derive("nar", reference, anchor)
