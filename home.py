import streamlit as st
import base64

# data_page = st.Page("pages/data exploration.py", title="", icon="🌍")
# model_page = st.Page("pages/modeling result.py", title="", icon="🚗")
# deploy_page = st.Page("pages/sentiment analyzer.py", title="", icon="🚀")

# pg = st.navigation([data_page, model_page, deploy_page])

# page configuration
st.set_page_config(page_title="Home | Sentiment Analysis BCA Mobile from Google PlayStore", layout="centered")
# pg.run()
st.title("Sentiment Analysis Final Project by :rainbow[Christopher Darren]")

st.header("What's this project about?")
st.write("This project revolves around sentiment analysis exploration project of mobile banking review app called BCA Mobile on Google PlayStore and a classifier model to classify text within the polarity of positive, negative or neutral text.")
st.write("To get start, choose what page do you want to see at sidebar of the page👈. For mobile _users_ see the sidebar on top of the page☝.")
st.header("🔍What can you do on this website?")
st.markdown("- **Data Exploration:** Explore visualization of BCA Mobile Review from Google PlayStore.")
st.markdown("- **Model Results:** See each model results visualization after training and evaluation.")
st.markdown("- **Sentiment Classification Analyzer:** Classify text using the Transformer Model (DistilBERT).")


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
