from pathlib import Path
import numpy as np
import joblib
import streamlit as st


st.set_page_config(page_title="Iris Flower Classification", page_icon="🌸", layout="wide")


@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "models" / "iris_svc_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan di: {model_path}")
    return joblib.load(model_path)


class_names = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]


try:
    model = load_model()
    model_ready = True
except Exception as e:
    model = None
    model_ready = False


st.title("🌸 Iris Flower Classification")
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
        st.warning(f"Model belum bisa dimuat: {e}")


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