import streamlit as st
import pandas as pd
import altair as alt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define models
models = {
    "ROBERTA": "cardiffnlp/twitter-roberta-base-sentiment",
}

# Sidebar "How to Use" message
how_to_use = """
**How to Use**
1. Enter text in the text area
2. Click the 'Analyze' button to get the predicted sentiment of the text
"""
st.sidebar.markdown(how_to_use)

# Main content
choice = st.sidebar.radio("Navigation", ["Home", "About"])

if choice == "Home":
    st.subheader("Tweet Sentiment Analyzer by ASVA")

    # Select model
    model_name = "ROBERTA"  # Only one model available

    # Text input and analyze button
    with st.form(key="nlpForm"):
        raw_text = st.text_area("Enter the Tweet")
        submit_button = st.form_submit_button(label="Analyze")

    # Display balloons on submit
    if submit_button:
        st.balloons()

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(models[model_name])
        model = AutoModelForSequenceClassification.from_pretrained(models[model_name])

        # Tokenize the input text
        inputs = tokenizer(raw_text, return_tensors="pt")

        # Make prediction
        outputs = model(**inputs)
        predicted_class = outputs.logits.argmax().item()
        score = outputs.logits.softmax(dim=1)[0][predicted_class].item()

        # Display sentiment
        sentiment_map = {2: "Positive", 1: "Neutral", 0: "Negative"}
        sentiment_emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}
        sentiment_label = sentiment_map[predicted_class]
        st.write(f"Predicted Sentiment: {sentiment_label} {sentiment_emoji[sentiment_label]}")

        # Create results DataFrame and chart
        data = {'Sentiment Class': ['Negative', 'Neutral', 'Positive'], 'Score': [0.0, 0.0, 0.0]}
        data['Score'][predicted_class] = score
        results_df = pd.DataFrame(data)
        chart = alt.Chart(results_df).mark_bar(width=50).encode(x="Sentiment Class", y="Score", color="Sentiment Class")
        st.altair_chart(chart, use_container_width=True)
        st.write(results_df)

else:
    st.subheader("About")
    st.write(
        "This is a sentiment analysis NLP app developed by Team ASVA for analyzing tweets."
    )
