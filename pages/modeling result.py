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