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
            st.markdown("### 6 Parameter Model Results")
            display_results(pipe_lr1, raw_text)

        with col2:
            st.markdown("### 13 Parameter Model Results")
            display_results(pipe_lr2, raw_text)

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
    proba_df = pd.DataFrame(probability, columns=model.classes_)
    proba_df_clean = proba_df.T.reset_index()
    proba_df_clean.columns = ["emotions", "probability"]

    fig = alt.Chart(proba_df_clean).mark_bar().encode(
        x='emotions', y='probability', color='emotions'
    )
    st.altair_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()