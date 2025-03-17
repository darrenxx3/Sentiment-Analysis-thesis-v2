import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
import plotly.express as px

# page configuration
st.set_page_config(layout="wide")
# Ui
st.title("Hello world")
st.markdown("---")

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