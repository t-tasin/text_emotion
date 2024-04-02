import streamlit as st
import joblib

# Load two models
pipe_lr1 = joblib.load(open("Model/text_emotion_6param.pkl", "rb"))
pipe_lr2 = joblib.load(open("Model/text_emotion.pkl", "rb"))

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", 
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", 
    "shame": "😳", "surprise": "😮"
}

def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text with Two Models")

    if st.button('Load Models'):
        st.write("Models loaded successfully!")
        st.write("Emotion-Emoji mapping available.")

if __name__ == '__main__':
    main()