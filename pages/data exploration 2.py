"""

data exploration 2.py

more visualization results
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import plotly.express as px
import base64
import matplotlib.pyplot as plt
import gensim
import gensim.corpora as corpora
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from st_wordcloud import st_wordcloud
from wordcloud import WordCloud
import re
import nltk
from collections import defaultdict


# page configuration
st.set_page_config(layout="wide", page_icon="💨💨")

# Ui
st.title("Exploration Data Analysis 2🌍")
st.markdown("---")

df = pd.read_csv("bca_preprocessed_data_stlit.csv", delimiter=',')

col1, col2 = st.columns([1.45,1], border=True)
col3, col4 = st.columns([1.2,1], border=True)

with col1:
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
    sentiment_per_topic = np.zeros((5,3)) # 5 topics array , 3 sentiment categories
 
    topic_counts = lda_model.get_document_topics(corpus, minimum_probability=0)
    topic_distributions = [0] * 5
    topic_mapping = {
    0: 2,  # Topic 0 -> "App Features & Performance"
    1: 3,  # Topic 1 -> "Update & Force Closes issues"
    2: 1,  # Topic 2 -> "Balance & Transactions"
    3: 4,  # Topic 3 -> "English Reviews"
    4: 0   # Topic 4 -> "Verification & Login issues"
    }

        # count sentiment per topic
    for i, doc in enumerate(topic_counts):
        if i < len(df):
            main_topic = max(doc, key=lambda x: x[1])[0] # Get most relevant topic
            mapped_topic = topic_mapping[main_topic] # Map to predefined label
            sentiment = int(df.iloc[i]["sentiment"])
            sentiment_per_topic[mapped_topic][sentiment] +=1

    topic_labels =[
    "Verification & Login issues",
    "Balance & Transactions",
    "App Features & Performance",
    "Update & Force Closes issues",
    "English Reviews"
    ]
    
    # plot sentiment labels and colors
    sentiment_labels = ["Negative","Neutral","Positive"]
    colors = ["red","yellow","green"]

    # plot sentiment breakdown per topic
    fig, ax = plt.subplots(figsize=(10,6))
    bar_width = 0.3

    bars = []
    for i in range(3):
        bars.append(ax.bar(
            np.arange(5) + i * bar_width,  # Added a comma here, and position bars
            sentiment_per_topic[:, i],
            bar_width,
            label=sentiment_labels[i],
            color=colors[i]
        ))

    ax.set_xlabel("Topics")
    ax.set_ylabel("Review Count")
    ax.set_title("BCA Mobile Sentiment Distribution accross Topics")
    ax.set_xticks(np.arange(5)+ bar_width)
    ax.set_xticklabels(topic_labels, rotation=30,ha="right")
    ax.legend()

    # add labels on top of each negative and positive bars
    for i in range(5):
        for j in [0,2]: # 0: negative 2: positive
            bar = bars[j][i]
            yval = bar.get_height()
            ax.text(bar.get_x()+ bar.get_width() / 2, yval + 1, int(yval), ha='center', va='bottom')

    # display plot
    st.pyplot(fig)

with col2:
    st.write("WordCloud")

    # making a side bar
    st.sidebar.header("Filters")

    df['at'] = pd.to_datetime(df['at'], errors="coerce") #convert datetime

    min_date = df['at'].min()
    max_date = df['at'].max()

    
    start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date])

    sentiment_option = {2:"Positive", 1:"Neutral", 0:"Negative"}
    selected_sentiments = st.sidebar.multiselect("Select Sentiment", options=sentiment_option.keys(),
                                                 format_func=lambda x: sentiment_option[x], default=list(sentiment_option.keys()))
    
    #convert start date and end date
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    #apply for filtering
    filtered_df = df[(df['at']>= start_date) & (df['at'] <= end_date)]
    filtered_df = filtered_df[filtered_df["sentiment"].isin(selected_sentiments)]

    st.sidebar.write(f"Filtered Reviews: {len(filtered_df)}")

    if not filtered_df.empty and "processedwords" in filtered_df.columns:
        all_words = []
        for words in df["processedwords"].dropna():
            if isinstance(words, list):
                all_words.extend(words)
            elif isinstance(words, str):
                all_words.extend(words.split())
        
        if all_words:
            word_counts = Counter(all_words)
            wordcloud = WordCloud(width=800, height=600, background_color="white", colormap="viridis", max_words=100).generate_from_frequencies(word_counts)

            fig, ax = plt.subplots(figsize=(10,5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        else:
            st.warning("No words found!")
    else:
        st.error("---")

with col3:
    st.subheader("convert to plotly.express later")
    # calculate word frequencies for each sentiment
    positive_words = df[df["sentiment"] == 2]['processedwords']
    negative_words = df[df["sentiment"] == 0]['processedwords']
    neutral_words = df[df["sentiment"] == 1]['processedwords']

    positive_frequen = Counter([word for words in positive_words for word in words])
    negative_frequen = Counter([word for words in negative_words for word in words])
    neutral_words_frequen = Counter([word for words in neutral_words  for word in words])

    # Get top 10 words from each sentiment
    top_n = 10
    top_positive = [word for word, _ in positive_frequen.most_common(top_n)]
    top_negative = [word for word, _ in negative_frequen.most_common(top_n)]

    #combine words and create DataFrame
    combined_words = list(set(top_positive+top_negative)) #merging
    bca_combined = pd.DataFrame({'word':combined_words})
    bca_combined['positive'] = bca_combined['word'].apply(lambda x: positive_frequen.get(x,0 ))
    bca_combined['negative'] = bca_combined['word'].apply(lambda x: negative_frequen.get(x,0))

    # Calculate absolute difference and Sort by importance
    bca_combined['diff'] = abs(bca_combined['positive'] - bca_combined['negative'])
    bca_sorted = bca_combined.nlargest(10, 'diff').sort_values('diff', ascending=True)

    #Plot the visualization
    plt.figure(figsize=(8,6))
    positive_bar = plt.barh(bca_sorted['word'], bca_sorted['positive'], color='green', label='Positive')
    negative_bar = plt.barh(bca_sorted['word'], -bca_sorted['negative'], color='red', label='Negative') # -bca sorted for minus negative

    for bar in positive_bar:
        plt.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2, #adjust position
                    str(int(bar.get_width())), va="center", fontsize=10, color="black")
            
    for bar in negative_bar:
        plt.text(bar.get_width() - 150, bar.get_y() + bar.get_height()/2, #adjust position
                    str(abs(int(bar.get_width()))), va="center", fontsize=10, color="black")

    plt.axvline(x=0, color='black', linewidth=1) #Add vertical line at 0

    # Set max width for centering
    plt.xlabel('Frequency')
    plt.ylabel("Words")
    plt.title('Top 10 most frequent words based on BCA Mobile Review data at Google Play Store')
    plt.legend()
    plt.grid(axis='x', linestyle='--')
    st.pyplot(plt)

with col4:
    st.subheader("Top 10 Positive & Negative on Update and Force Closes Topic issues")

    word_counts = {topic: {"positive": Counter(), "negative": Counter()} for topic in topic_mapping.values()}

        # Assign topics and count words per sentiment
    for i, doc in enumerate(topic_counts):
        if i < len(df):  # Ensure index is within range
            main_topic = max(doc, key=lambda x: x[1])[0]  # Get most relevant topic
            mapped_topic = topic_mapping[main_topic]  # Map to predefined label
            sentiment = int(df.iloc[i]["sentiment"])
            
            if sentiment in [0, 2]:  # Only negative (0) and positive (2)
                sentiment_label = "positive" if sentiment == 2 else "negative"
                words = df.iloc[i]["processedwords"]  # Use preprocessed words
                word_counts[mapped_topic][sentiment_label].update(words)

    # Define the topic name we want to visualize
    selected_topic_name = "Update & Force Closes issues"

    # Find the correct topic index from the mapping
    mapped_index = None
    for key, value in topic_mapping.items():
        if value == 4:  # "Update & Force Closes issues" is mapped to index 3
            mapped_index = key
            break

    if mapped_index is not None and mapped_index in word_counts:
        sentiments = word_counts[mapped_index]  # Get word counts for the correct topic

        if "positive" in sentiments and "negative" in sentiments:
            # Get top words for positive and negative sentiment
            top_positive = sentiments["positive"].most_common(10)
            top_negative = sentiments["negative"].most_common(10)

            # Extract words and counts
            positive_words, positive_counts = zip(*top_positive) if top_positive else ([], [])
            negative_words, negative_counts = zip(*top_negative) if top_negative else ([], [])

            # Ensure equal length by padding with zeros (if necessary)
            max_len = max(len(positive_words), len(negative_words))
            positive_words += ("",) * (max_len - len(positive_words))
            negative_words += ("",) * (max_len - len(negative_words))
            positive_counts += (0,) * (max_len - len(positive_counts))
            negative_counts += (0,) * (max_len - len(negative_counts))

            # Reverse order for better visualization
            positive_words = positive_words[::-1]
            negative_words = negative_words[::-1]
            positive_counts = positive_counts[::-1]
            negative_counts = negative_counts[::-1]

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))
            fig.suptitle(f"Top 10 Positive and Negative Words for {selected_topic_name} Topic", fontsize=14)

            # Define y-axis positions
            y_pos = np.arange(max_len)

            # Set max width for centering
            max_count = max(max(positive_counts, default=0), max(negative_counts, default=0))

            # Plot bars with centering
            ax.barh(y_pos, [-val for val in negative_counts], color="red", label="Negative Words", align="center")
            ax.barh(y_pos, positive_counts, color="green", label="Positive Words", align="center")

            # Set labels and titles
            ax.set_yticks(y_pos)
            ax.set_yticklabels(positive_words)  # Display words on the left side
            ax.set_xlabel("Word Frequency")
            ax.set_xlim(-max_count, max_count)  # Ensure both sides are symmetric
            ax.axvline(0, color="black", linewidth=1)  # Vertical center line at 0
            ax.legend(loc="upper right")

            # Add labels on bars
            for i, (neg, pos) in enumerate(zip(negative_counts, positive_counts)):
                if neg > 0:
                    ax.text(-neg, y_pos[i], f"{neg}", ha="right", va="center", color="white",
                            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1))
                if pos > 0:
                    ax.text(pos, y_pos[i], f"{pos}", ha="left", va="center", color="white",
                            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1))

            # plt.gca().invert_yaxis()  # Invert to keep the highest frequency at the top
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning(f"No sentiment data found for topic {selected_topic_name}.")
    else:
        st.warning(f"Topic '{selected_topic_name}' not found in word_counts or mapping.")

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