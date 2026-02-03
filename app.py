import streamlit as st
import requests
import time

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Bone AI", layout="wide")
st.title("🦴 Система анализа переломов")

with st.sidebar:
    st.header("Настройки")
    conf_val = st.slider("Порог уверенности (Conf)", 0.01, 1.0, 0.25)
    model_val = st.selectbox("Модель", ["fast", "accurate"])

uploaded_file = st.file_uploader("Загрузите рентгеновский снимок", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Оригинал")
        st.image(uploaded_file, use_container_width=True)
    
    if st.button("🚀 Запустить нейросеть", use_container_width=True):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        data = {"model_type": model_val, "imgsz_w": 640, "imgsz_h": 640, "conf": conf_val}
        
        with col2:
            with st.spinner("Анализ снимка..."):
                try:
                    res = requests.post(f"{BACKEND_URL}/predict", files=files, data=data).json()
                    t_id = res.get("task_id")
                    while True:
                        s_res = requests.get(f"{BACKEND_URL}/status/{t_id}").json()
                        if s_res.get("status") == "completed":
                            count = s_res.get("count", 0)
                            if count > 0:
                                st.image(f"data:image/jpeg;base64,{s_res['image']}", use_container_width=True)
                                st.error(f"⚠️ Найдено патологий: {count}")
                            else:
                                st.balloons()
                                st.success("✅ Всё в порядке! Переломов не обнаружено.")
                            break
                        elif s_res.get("status") == "error":
                            st.error(f"Ошибка: {s_res.get('message')}")
                            break
                        time.sleep(0.5)
                except Exception as e:
                    st.error(f"Не удалось подключиться к бэкенду (main.py). Ошибка: {e}")