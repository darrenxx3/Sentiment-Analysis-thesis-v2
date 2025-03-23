import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
import plotly.express as px


# page configuration
st.set_page_config(layout="wide", page_icon="💨💨")

# Ui
st.title("Exploration Data Analysis 2🌍")
st.markdown("---")

df = pd.read_csv("bca_preprocessed_data_stlit.csv", delimiter=',')
