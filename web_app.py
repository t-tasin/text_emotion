import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Load two models
pipe_lr1 = joblib.load(open("Model/text_emotion.pkl", "rb"))
pipe_lr2 = joblib.load(open("Model/text_emotion_6param.pkl", "rb"))

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", 
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", 
    "shame": "😳", "surprise": "😮"
}

def predict_emotions(model, docx):
    results = model.predict([docx])
    return results[0]

def get_prediction_proba(model, docx):
    results = model.predict_proba([docx])
    return results

def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text with Two Models")

    with st.form(key='emotion_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Model 1 Results")
            display_results(pipe_lr1, raw_text)

        with col2:
            st.markdown("### Model 2 Results")
            display_results(pipe_lr2, raw_text)

def display_results(model, text):
    prediction = predict_emotions(model, text)
    probability = get_prediction_proba(model, text)

    st.success("Original Text")
    st.write(text)

    st.success("Prediction")
    # Use the model's classes to get the appropriate emoji
    emoji_icon = emotions_emoji_dict.get(prediction, "")
    st.write(f"{prediction}: {emoji_icon}")
    st.write(f"Confidence: {np.max(probability)}")

    st.success("Prediction Probability")
    # Adjust the DataFrame creation to use the model's classes directly
    proba_df = pd.DataFrame(probability, columns=model.classes_)
    proba_df_clean = proba_df.T.reset_index()
    proba_df_clean.columns = ["emotions", "probability"]

    fig = alt.Chart(proba_df_clean).mark_bar().encode(
        x='emotions', y='probability', color='emotions'
    )
    st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
