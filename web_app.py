import streamlit as st
import joblib

# Load two models
pipe_lr1 = joblib.load(open("Model/text_emotion_6param.pkl", "rb"))
pipe_lr2 = joblib.load(open("Model/text_emotion.pkl", "rb"))

def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text with Two Models")

    if st.button('Load Models'):
        st.write("Models loaded successfully!")

if __name__ == '__main__':
    main()
