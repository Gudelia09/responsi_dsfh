from pathlib import Path

import numpy as np
import streamlit as st

try:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    sklearn_available = True
    sklearn_error = None
except ModuleNotFoundError as err:
    sklearn_available = False
    sklearn_error = err

try:
    import joblib
    joblib_available = True
except ModuleNotFoundError:
    joblib = None
    joblib_available = False


st.set_page_config(page_title="Iris Flower Classification", page_icon="🌸", layout="wide")


@st.cache_resource
def load_model():
    if not sklearn_available:
        raise ModuleNotFoundError(
            "Paket scikit-learn belum terinstal. Jalankan 'pip install -r requirements.txt' terlebih dahulu."
        )
    if not joblib_available:
        raise ModuleNotFoundError(
            "Paket joblib belum terinstal. Jalankan 'pip install -r requirements.txt' terlebih dahulu."
        )

    model_path = Path(__file__).resolve().parent / "models" / "iris_svc_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception:
            model_path.unlink(missing_ok=True)

    iris = load_iris()
    X_train, _, y_train, _ = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model


class_names = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]


try:
    model = load_model()
    model_ready = True
    load_error = None
except Exception as e:
    model = None
    model_ready = False
    load_error = e


st.title("🌸 Iris Flower Classification")
if not sklearn_available:
    st.error(
        "scikit-learn tidak tersedia di lingkungan saat ini. Pastikan dependency terinstal: `pip install -r requirements.txt`"
    )
    if sklearn_error is not None:
        st.write(f"Detail error: {sklearn_error}")
elif not joblib_available:
    st.error(
        "joblib tidak tersedia di lingkungan saat ini. Pastikan dependency terinstal: `pip install -r requirements.txt`"
    )
    if load_error is not None:
        st.write(f"Detail error: {load_error}")
else:
    st.caption("Aplikasi prediksi spesies bunga Iris dengan model SVM")

st.write("Masukkan nilai fitur bunga Iris untuk melihat prediksi spesies yang paling mungkin.")

with st.sidebar:
    st.header("Input Fitur")
    st.caption("Sesuaikan nilai berikut lalu klik tombol prediksi.")

    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.5, 5.1, 0.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.5, 0.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 3.0, 0.2, 0.1)

    st.divider()
    if model_ready:
        st.success("Model siap digunakan")
    else:
        st.warning(f"Model belum bisa dimuat: {load_error}")


features = np.array([[sepal_length, sepal_width, petal_length, petal_width]], dtype=float)

col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.subheader("Input Saat Ini")
    st.table(
        {
            "Fitur": [
                "Sepal Length",
                "Sepal Width",
                "Petal Length",
                "Petal Width",
            ],
            "Nilai": [sepal_length, sepal_width, petal_length, petal_width],
        }
    )

with col2:
    st.subheader("Hasil Prediksi")

    if st.button("Prediksi", use_container_width=True):
        if not model_ready:
            st.error("Model belum tersedia. Periksa file model dan dependensi aplikasi.")
        else:
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]

            st.success(f"Prediksi: {class_names[prediction]}")
            st.progress(float(np.max(probability)))

            st.write("Probabilitas per kelas:")
            for name, prob in zip(class_names, probability):
                st.write(f"- **{name}**: {prob:.4f}")