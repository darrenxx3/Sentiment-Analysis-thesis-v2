"""

data exploration.py

A code for displaying visualization entry
"""

import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
import plotly.express as px
import base64
import matplotlib.pyplot as plt
import gensim
import gensim.corpora as corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk
from collections import defaultdict


# page configuration
st.set_page_config(layout="wide", page_icon="💨")

# # adding background image
# def set_bg_image(main_bg):
#     '''
#     A function to unpack an image from root folder and set as bg.
 
#     Returns
#     -------
#     The background.
#     '''
#     # set background image name
#     main_bg_ext = "png"
        
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()}) no-repeat center center fixed;
#              background-size: cover;
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )
# set_bg_image("./img/vecteezy_businessman.jpg")

# Ui
st.title("Exploration Data Analysis🌐")
st.markdown("---")

#load dataset
df = pd.read_csv("bca_preprocessed_data_stlit.csv", delimiter=',')
df= df.dropna()
df = df.rename(columns={"rating": "score"})

#set columns
col1, col2 = st.columns(2, border=True)
col3, col4 = st.columns([1.3,1], border=True)

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
                    title="BCA Mobile PlayStore Reviews Count applied with Rule-based technique")
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

    # creating the filter dropdown
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
                    title="Sentiment Trend Over time from 2022 - 2025 applied with Ruled-based technique")
    st.altair_chart(chart, use_container_width=True)
    
with col4:
    #ensure NLTK downloaded
    nltk.download("punkt")
    nltk.download("stopwords")

    df = pd.read_csv("bca_preprocessed_data_stlit.csv", delimiter=',')
    df = df.dropna(subset=["content"])

    stop_words_indo = set(stopwords.words("indonesian"))
    stop_words_en = set(stopwords.words("english"))
    custom_stopwords = {"nya", "yg", "gak", "ga", "sih", "dong", "deh", "nih", "aja", "bgt","saya", "dan","sdh","sudah"
                    ,"sy","lg","tdk", "udah","lagi"}

    stop_words = stop_words_indo.union(stop_words_en, custom_stopwords)

    def preprocess_text(text):
        text = text.lower() #c onvert to lowercase
        text = re.sub(r'\d+', '',text) # remove numbers
        text = re.sub(r'[^\w\s]','',text) # remove special characters
        text = re.sub(r'(.)\1+', r'\1\1', text) #limit repeating characters
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        return tokens

    df["processedwords"] = df["content"].apply(preprocess_text)

    dictionary = corpora.Dictionary(df["processedwords"])
    corpus = [dictionary.doc2bow(text)for text in df["processedwords"]]
    
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                   id2word=dictionary,
                                   num_topics=5,
                                   random_state=42,
                                   passes=10)

    # LDA results 
    topic_counts = lda_model.get_document_topics(corpus, minimum_probability=0)
    topic_distributions = [0] * 5
    topic_mapping = {
    0: 2,  # Topic 0 -> "App Features & Performance"
    1: 3,  # Topic 1 -> "Update & Force Closes issues"
    2: 1,  # Topic 2 -> "Balance & Transactions"
    3: 4,  # Topic 3 -> "English Reviews"
    4: 0   # Topic 4 -> "Verification & Login issues"
    } 

    for doc in topic_counts:
        main_topic = max(doc, key=lambda x: x[1])[0] #get most relevant topic
        mapped_topic = topic_mapping[main_topic] #map predefined labels
        topic_distributions[mapped_topic] += 1
    
    topic_labels =[
    "Verification & Login issues",
    "Balance & Transactions",
    "App Features & Performance",
    "Update & Force Closes issues",
    "English Reviews"
    ]
    
    # convert to dataframe for streamlit
    topic_df = pd.DataFrame({"Topic": topic_labels, "Count":topic_distributions})

    #creating the plot requirements
    fig = px.bar(
        topic_df, x="Topic", y="Count", text="Count", color="Topic", title="Topic Distributions on BCA Mobile Reviews",
        labels={"Count": "Review Count"},
        template="plotly_white"
    )
    fig.update_traces(textposition="outside")

    # display plot
    st.plotly_chart(fig)

    # st.subheader("Topics")
    # topics = lda_model.print_topics(num_words=10)
    # for i, topic in enumerate(topics):
    #     st.write(f"**Topic {i+1}:** {topic[1]}")


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