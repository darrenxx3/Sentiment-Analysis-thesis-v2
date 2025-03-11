"""
main.py

This is the main entry point for the sentiment analysis application.
"""
from io import StringIO
import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import altair as alt
import base64
import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings
from utils import load_model

# adding background image
def set_bg_image(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()}) no-repeat center center fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
set_bg_image("./img/senti_bg.jpg")

# Store for adding pictures
Sentiment_picture = {
    0: "./img/sad.jpg",
    1: "./img/netralface.jpeg",
    2: "./img/smile.jpg"
}

# load the trained model
model, tokenizer = load_model()

# Streamlit UI
st.title("Sentiment Analyzer Test📈")
st.subheader("Hey, use this sentiment analyzer! It's easy to use", divider="blue")

#user input
text = st.text_input(label="", placeholder="Type here")

col1, col2 = st.columns([0.9, 1]) #creating columns of 2
with col1:
    if st.button("Predict"):
        if text:   
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs =model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                sentiment = torch.argmax(probs, dim=-1).item()
                # score = outputs.logits.softmax(dim=1)[0][sentiment].item()

                # Compute the scores for all sentiments
                positive_score = probs[0][2].item()
                negative_score = probs[0][0].item()
                neutral_score = probs[0][1].item()

                # Compute the confidence level
                confidence_level = probs[0][sentiment].item()

                sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
                st.write("Results:")
                st.write(f"Sentiment: **{sentiment_labels[sentiment]}** , Confidence Score: **{confidence_level:.5f}**")

                # Display image corresponding emoji image
                image_loc = Sentiment_picture[sentiment]
                st.image(image_loc, width=250)
            
            with col2:
                alt_viz = pd.DataFrame({    
                    "Sentiment Class":["Negative","Neutral","Positive"],
                    "Score":[float(negative_score), float(neutral_score), float(positive_score)]
                })

                # Create bar
                chart = alt.Chart(alt_viz).mark_bar().encode(
                    x=alt.X("Sentiment Class", sort=["Negative","Neutral","Positive"]),
                    y="Score",
                    color="Sentiment Class"
                )
                st.altair_chart(chart, use_container_width=True)
                st.write(alt_viz) # add table below the altair chart
                
        else:
            st.warning("Please enter text before predicting.😁")

# Using uploaded file option
st.subheader("Option 2: Via Upload CSV file for sentiment analysis🙏", divider="red")

col1, col2 = st.columns([0.9, 1]) #creating columns of 2
uploaded_file = st.file_uploader("Choose a file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'content' not in df.columns:
        st.error("The file must have 'content' column")
    else:
        def predict_sentiment(text):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs =model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                sentiment = torch.argmax(probs, dim=-1).item()
                sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
                confidence_level = probs[0][sentiment].item()
            return confidence_level, sentiment_labels[sentiment] 

        df[['Confidence', 'Predicted Sentiment']] = df['content'].apply(lambda x: pd.Series(predict_sentiment(str(x))))

        # Display Results
        st.subheader("Results")
        st.dataframe(df, use_container_width=True)
            
        #download results
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "bca_classification.csv", "text/csv")

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