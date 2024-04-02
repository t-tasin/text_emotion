import streamlit as st
import joblib

# Load two models
pipe_lr1 = joblib.load(open("Model/text_emotion_6param.pkl", "rb"))
pipe_lr2 = joblib.load(open("Model/text_emotion.pkl", "rb"))

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
            st.write("Results from Model 1 will be displayed here.")
        with col2:
            st.markdown("### Model 2 Results")
            st.write("Results from Model 2 will be displayed here.")

if __name__ == '__main__':
    main()