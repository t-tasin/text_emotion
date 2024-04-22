# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load two models
pipe_lr1 = joblib.load(open("Model/text_emotion_6param.pkl", "rb"))
pipe_lr2 = joblib.load(open("Model/text_emotion.pkl", "rb"))
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", 
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", 
    "shame": "😳", "surprise": "😮"
}

# Function to predict the dominant emotion in a given text
def predict_emotions(model, docx):
    results = model.predict([docx])
    return results[0]

# Function to get the prediction probabilities for each emotion
def get_prediction_proba(model, docx):
    results = model.predict_proba([docx])
    return results

def main():
    st.title("Text Emotion Detection") #Title
    st.subheader("Detect Emotions In Text with Two Models") #Sub Header

    # Streamlit form for user input
    with st.form(key='emotion_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    # Display results if text is submitted
    if submit_text:
        col1, col2 = st.columns(2) #Creating two column for displaying results

        #Display result from first model (6 Parameter Model)
        with col1:
            st.markdown("### 6 Parameter Model Results")
            display_results(pipe_lr1, raw_text)

        #Disuplay result from second model (6 Parameter Model)
        with col2:
            st.markdown("### 13 Parameter Model Results")
            display_results(pipe_lr2, raw_text)

# Helper function to display prediction results and probabilities
def display_results(model, text):
    prediction = predict_emotions(model, text)
    probability = get_prediction_proba(model, text)

    st.success("Original Text")
    st.write(text)

    st.success("Prediction")
    emoji_icon = emotions_emoji_dict.get(prediction, "")
    st.write(f"{prediction}: {emoji_icon}")
    
    # Calculate and format the confidence as a percentage
    confidence_fraction = np.max(probability)
    confidence_percentage = confidence_fraction * 100
    st.write(f"Confidence: {confidence_fraction:.4f} ({confidence_percentage:.2f}%)")

    st.success("Prediction Probability")

    # Create a DataFrame for displaying prediction probabilities
    proba_df = pd.DataFrame(probability, columns=model.classes_)
    proba_df_clean = proba_df.T.reset_index()
    proba_df_clean.columns = ["emotions", "probability"]

    # Create a bar chart for the probabilities using Altair
    fig = alt.Chart(proba_df_clean).mark_bar().encode(
        x='emotions', y='probability', color='emotions'
    )
    st.altair_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()