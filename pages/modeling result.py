import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Training & Evaluation Result🚅")

st.header("LSTM Result⏲")
col1, col2, col3 = st.columns(3)

with col1:
    st.image("https://i.redd.it/v75t68xjxmi51.jpg")
    st.subheader("Classification Report")
with col2:
    st.image("https://api.duniagames.co.id/api/content/upload/file/5672193221657192439.jpg")
    st.subheader("Confusion Matrix")

with col3:
    st.image("https://assets.ggwp.id/2023/06/fakta-menarik-Ichinose-Chizuru-featured-640x360.jpg")
    st.subheader("Train and Validation Loss")
st.markdown("---") #divider

st.header("DistilBERT Result")
col1, col2, col3 = st.columns(3)
with col1:
    st.image("https://i.redd.it/v75t68xjxmi51.jpg")
    st.subheader("Classification Report")

with col2:
    st.image("https://api.duniagames.co.id/api/content/upload/file/5672193221657192439.jpg")
    st.subheader("Confusion Matrix")

with col3:
    st.image("https://assets.ggwp.id/2023/06/fakta-menarik-Ichinose-Chizuru-featured-640x360.jpg")
    st.subheader("Train and Validation Loss")
st.markdown("---") #divider

st.header("DistilBERT+Optuna Result🎯")
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
with col1:
    st.image("https://i.redd.it/v75t68xjxmi51.jpg")
    st.subheader("Hyperparameter Importance")

with col2:
    st.image("https://api.duniagames.co.id/api/content/upload/file/5672193221657192439.jpg")
    st.subheader("Classification Report")

with col3:
    st.image("https://assets.ggwp.id/2023/06/fakta-menarik-Ichinose-Chizuru-featured-640x360.jpg")
    st.subheader("Confusion Matrix")

with col4:
    st.image("https://doublesama.com/wp-content/uploads/2022/08/Kanokari-2nd-Season-Episode-6-Chizuru-on-her-balcony.jpg")
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