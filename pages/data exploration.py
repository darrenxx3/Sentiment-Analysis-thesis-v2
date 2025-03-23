"""

data exploration.py

A code for displaying visualization entry
"""

import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
import plotly.express as px


# page configuration
st.set_page_config(layout="wide", page_icon="💨")

# Ui
st.title("Exploration Data Analysis🌐")
st.markdown("---")

#load dataset
df = pd.read_csv("bca_preprocessed_data_stlit.csv", delimiter=',')
df = df.rename(columns={"rating": "score"})

#set columns
col1, col2 = st.columns(2, border=True)
col3, col4 = st.columns([1.5,1], border=True)

with col1:
    score_counts  = df["score"].value_counts().sort_index()
    score_df = pd.DataFrame({"Star Rating": score_counts.index, "Number of Reviews": score_counts.values})
    
    chart = alt.Chart(score_df).mark_bar().encode(
                    x=alt.X('Star Rating:O', title='Star Rating', axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('Number of Reviews:Q', title='Number of Reviews'),
                    color=alt.Color('Star Rating:N', legend=alt.Legend(title="Rating"))).properties(
                    title="BCA Mobile PlayStore Reviews Count")
    st.altair_chart(chart, use_container_width=True)

with col2:
    score_counts  = df["sentiment"].value_counts().sort_index()
    score_df = pd.DataFrame({"Star Rating": score_counts.index, "Number of Reviews": score_counts.values})
    sentiment_map = {0: "Negative", 1:"Neutral", 2:"Positive"} #translate and mapping labels
    score_df["Sentiment Label"] = score_df["Star Rating"].map(sentiment_map) 
    
    chart = alt.Chart(score_df).mark_bar().encode(
                    x=alt.X('Sentiment Label:N', title='Sentiment Label', axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('Number of Reviews:Q', title='Number of Reviews'),
                    color=alt.Color('Star Rating:N', legend=alt.Legend(title="Sentiment label"))).properties(
                    title="BCA Mobile PlayStore Reviews Count applied with LDA technique")
    st.altair_chart(chart, use_container_width=True)

with col3:
    st.subheader("Sentiment Trend📊")
    
    df['at'] = pd.to_datetime(df['at'], errors="coerce")
    df = df.dropna(subset=['at'])
    #df['Month-Year'] = df ['at'].dt.strftime("%Y-%m") # convert datetime
    df['Month-Year'] = df ['at'].dt.to_period("M").astype(str) # convert datetime

    df = df.sort_values(by='at')

    sentiment_line = df.groupby(["Month-Year","sentiment"]).size().reset_index(name="Count")

    sentiment_map = {0: "Negative", 1:"Neutral", 2:"Positive"} #translate and mapping labels
    sentiment_line["Sentiment"] = sentiment_line["sentiment"].map(sentiment_map)

    sentiment_option = ["All"] + list(sentiment_map.values()) # No filter added
    selected_sentiment = st.selectbox("Select:", sentiment_option) #creating the select box

    if selected_sentiment == "All":
        filter_data = sentiment_line
    else:
        filter_data = sentiment_line[sentiment_line["Sentiment"] == selected_sentiment]

    chart = alt.Chart(filter_data).mark_line(point=True).encode(
        x=alt.X('Month-Year:O', title='Month Year', axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('Count:Q', title='Number of Reviews'),
                    color=alt.Color('Sentiment:N', legend=alt.Legend(title="Sentiment label Type"))).properties(
                    title="Sentiment Trend Over time from 2022 - 2025 applied with LDA technique")
    st.altair_chart(chart, use_container_width=True)
    
with col4:
    from wordcloud import WordCloud
    
    st.subheader("WordCloud")


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