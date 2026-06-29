from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit_app


def test_load_model_creates_default_model_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(streamlit_app, "__file__", str(tmp_path / "streamlit_app.py"))
    streamlit_app.load_model.clear()

    model = streamlit_app.load_model()
    model_path = tmp_path / "models" / "iris_svc_model.pkl"

    assert model is not None
    assert model_path.exists()

    prediction = model.predict(np.array([[5.1, 3.5, 1.4, 0.2]], dtype=float))[0]
    assert isinstance(prediction, (int, np.integer))
