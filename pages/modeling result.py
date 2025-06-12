"""

modeling result.py

a result for see trained model, and evaluation including confusion matrix, train loss vis
"""

import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
import plotly.express as px

st.set_page_config(layout="wide", page_icon="🤖")
st.title("Training & Evaluation Result🚅")

st.header("LSTM Result⏲")
col1, col2, col3 = st.columns(3, border=True, vertical_alignment="center")

with col1:
    st.image("img/lstm_clacreport1.png")
    st.subheader("Classification Report")
with col2:
    st.image("img/lstm_conf.png")
    st.subheader("Confusion Matrix")

with col3:
    st.image("img/lstm_train_loss1.png")
    st.subheader("Train and Validation Loss")
st.markdown("---") #divider

st.header("DistilBERT Result")
col1, col2, col3 = st.columns(3, border=True, vertical_alignment="center")
with col1:
    st.image("img/distilbert_clacreport1.png")
    st.subheader("Classification Report")
    st.write("- Train result: 90.98%")
    st.write("- Evaluation result: 91.79%")

with col2:
    st.image("img/distilbert_conf.png")
    st.subheader("Confusion Matrix")

with col3:
    st.image("img/distilbert_train_loss1.png")
    st.subheader("Train and Validation Loss")
st.markdown("---") #divider

st.header("DistilBERT+Optuna Result🎯")
col1, col2 = st.columns(2, border=True)
col3, col4 = st.columns(2, border=True)
with col1:
    st.image("img/optuna_hyperparameter.png")
    st.subheader("Hyperparameter Importance")
    st.write("- Learning Rate memiliki value sebesar 0.26 yang menggambarkan pemilihan nilai learning rate yang optimal. Apabila learning rate terlalu besar dapat menyebabkan model tidak stabil sedangkan terlalu kecil dapat memperlambat proses training\n.")
    st.write("- Weight_decay juga memiliki pengaruh signifikan dengan nilai 0.25 berperan dalam regularisasi model untuk menghindari model *overfitting*.")
    st.write("- Gradient_accumulation_steps dan batch_size memiliki pengaruh yang cukup besar terhadap performa model.")
    st.write("- Max_grad_norm dengan nilai 0.11 memiliki pengaruh yang lebih kecil, tetapi tetap berperan dalam mencegah exploding gradients melalui teknik gradient clipping yang pada intinya untuk menstabilkan proses training.")
    st.write("- num_train_epoch menunjukkan pengaruh yang tidak signifikan sehingga bisa disimpulkan bahwa jumlah epoch cukup dengan jumlah yang kecil sehingga dengan menambahkan jumlah epoch tidak memberikan dampak yang besar terhadap performa model.")

with col2:
    st.image("img/optuna_clacreport.png")
    st.subheader("Classification Report")

with col3:
    st.image("img/optuna_conf.png")
    st.subheader("Confusion Matrix")

with col4:
    st.image("img/optuna_train_loss.png")
    st.subheader("Train and Validation Loss")

# page footer
footer ="""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f0f0f0; 
    color: #333; /*text color*/
    text-align: center;
    padding: 5px 0;
}
</style>
<div class="footer">
    <p><b>Copyright © 2025</b> Made in 💘 by <b>Christopher Darren</b>. All rights reserved.</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)