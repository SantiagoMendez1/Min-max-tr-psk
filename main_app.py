import streamlit as st
from ml_model import DataAnalyzer
from PIL import Image


st.set_page_config('Homepage', page_icon="🏡")

with st.sidebar:
    st.markdown("# Hello 👋")
    st.markdown("## Welcome to the Inventory Forecasting APP's")

st.title('Inventory Forecasting')

st.markdown("This application gives an approximate forecast of the minimum and maximum inventory for a product")

analyzer = DataAnalyzer('/data/stock_move.csv')

data = analyzer.pre_processing()

filter_products = analyzer.model_hyperparam_tuning(run=False)

selected_product = st.selectbox("Seleccione el producto", filter_products)

months_to_predict = st.selectbox("Seleccione los meses a predecir", [1, 2, 3, 4, 5, 6])

if st.button("Mostrar Predicción"):
    prediction = analyzer.prediction(selected_product, months_to_predict)
    plot_test = analyzer.plot_test()
    plot_pred = analyzer.plot_months_to_predict()
    image_path_test =  f'/Users/Camilo/Documents/presik/DB_MINMAXTODOREP/model/rf_prediction_{str(selected_product)}_test.png'
    st.markdown("Test Data Prediction Plot")
    image_test = Image.open(image_path_test)
    st.image(image_test, caption=f'Test data prediction for the product {selected_product}', use_column_width=True)
    image_path_pred =  f'/Users/Camilo/Documents/presik/DB_MINMAXTODOREP/model/rf_prediction_{str(selected_product)}_{str(months_to_predict)}_months.png'
    st.markdown("Prediction Plot")
    image_pred = Image.open(image_path_pred)
    st.image(image_pred, caption=f'Prediction for the product {selected_product}', use_column_width=True)
    st.markdown("Prediction Data")
    st.write(prediction)

