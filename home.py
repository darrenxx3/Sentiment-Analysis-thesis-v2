import streamlit as st


# data_page = st.Page("pages/data exploration.py", title="", icon="🌍")
# model_page = st.Page("pages/modeling result.py", title="", icon="🚗")
# deploy_page = st.Page("pages/sentiment analyzer.py", title="", icon="🚀")

# pg = st.navigation([data_page, model_page, deploy_page])

# page configuration
st.set_page_config(page_title="Home | Sentiment Analysis BCA Mobile from Google PlayStore", layout="centered")
# pg.run()
st.title("Sentiment Analysis Final Project by :rainbow[Christopher Darren]")

st.header("What's this project about?")
st.write("This project revolves around sentiment analysis exploration project of mobile banking review app called BCA Mobile on Google PlayStore.")
st.header("🔍What can you see?")
st.markdown("- **Data Exploration:** Explore visualization of BCA Mobile Review from Google PlayStore.")
st.markdown("- **Model Results:** See each model results graph after training.")
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
