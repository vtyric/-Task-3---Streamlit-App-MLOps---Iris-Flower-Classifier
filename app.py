import pickle

import numpy as np
import streamlit as st

flowers_images = {
    "Setosa": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Irissetosa1.jpg/413px-Irissetosa1.jpg",
    "Versicolor": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/800px-Iris_versicolor_3.jpg",
    "Virginica": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Iris_virginica_2.jpg/330px-Iris_virginica_2.jpg"
}

# Загрузка обученной модели
with open("model/IrisClassifier.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Классификация ирисов")

st.write("Введите параметры цветка для классификации:")

sepal_length = st.number_input("Длина чашелистика", min_value=0.0, value=5.0)
sepal_width = st.number_input("Ширина чашелистика", min_value=0.0, value=3.5)
petal_length = st.number_input("Длина лепестка", min_value=0.0, value=1.5)
petal_width = st.number_input("Ширина лепестка", min_value=0.0, value=0.2)

if st.button("Предсказать"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    pred_class = prediction[0]

    st.success(f"Предсказанный класс ириса: {pred_class}")

    if pred_class in flowers_images:
        st.image(flowers_images[pred_class], caption=pred_class)
    else:
        st.warning("Изображение для данного класса не найдено.")
