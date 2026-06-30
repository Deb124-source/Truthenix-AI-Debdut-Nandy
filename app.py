import streamlit as st
import pickle


# ----------------------------
# Load Model & Vectorizer
# ----------------------------

model = pickle.load(
    open("fake_news_model.pkl","rb")
)

vectorizer = pickle.load(
    open("vectorizer.pkl","rb")
)



# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Truthenix AI",
    page_icon="📰",
    layout="centered"
)


# ----------------------------
# Header
# ----------------------------

st.title("📰 Truthenix AI")
st.subheader(
    "AI Powered Fake News Detector"
)

st.write(
    "Analyze a news article and detect whether it is REAL or FAKE using Machine Learning."
)



# ----------------------------
# Input
# ----------------------------

news_text = st.text_area(
    "Enter News Article Text",
    height=200,
    placeholder="Paste the news content here..."
)



# ----------------------------
# Prediction
# ----------------------------

if st.button("🔍 Analyze News"):


    if news_text.strip()=="":
        st.warning(
            "Please enter a news article."
        )


    else:

        # Convert text into TF-IDF
        transformed_text = vectorizer.transform(
            [news_text]
        )


        prediction = model.predict(
            transformed_text
        )


        # Probability (confidence)
        confidence = None

        if hasattr(model,"decision_function"):

            score = model.decision_function(
                transformed_text
            )[0]

            confidence = abs(score)



        # Result

        if prediction[0] == 0:

            st.error(
                "🚨 FAKE NEWS DETECTED"
            )

            st.write(
                "The model predicts this article may contain misleading information."
            )


        else:

            st.success(
                "✅ REAL NEWS DETECTED"
            )

            st.write(
                "The model predicts this article is likely authentic."
            )


        if confidence:

            st.info(
                f"Model Confidence Score: {confidence:.2f}"
            )



# ----------------------------
# About Section
# ----------------------------

st.divider()

st.caption(
"""
Truthenix uses:
- TF-IDF Text Vectorization
- Passive Aggressive Classifier
- Natural Language Processing

Built for AI-based fake news detection.
"""
)
