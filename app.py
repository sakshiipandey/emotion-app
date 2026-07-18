import streamlit as st
import joblib
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="EmotionSense AI",
    page_icon="🤖",
    layout="centered"
)

# ---------------- LOAD LOGO ---------------- #

logo = Image.open("assets/emotion-recognition.png")

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#4F46E5,#7C3AED,#EC4899);
}

.main-card{
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(15px);
    padding:30px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.2);
}

.title{
    text-align:center;
    color:white;
    font-size:50px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:white;
    font-size:18px;
    margin-bottom:25px;
}

.stTextArea textarea{
    border-radius:15px;
    font-size:18px;
}

div.stButton > button{
    width:100%;
    height:55px;
    border:none;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
    background:#4F46E5;
    color:white;
}

div.stButton > button:hover{
    background:#312E81;
    color:white;
}

.result{
    background:white;
    border-radius:18px;
    padding:25px;
    margin-top:20px;
    text-align:center;
}

.footer{
    text-align:center;
    color:white;
    padding-top:20px;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- EMOTION MAP ---------------- #

emotion_map = {
    0: ("Sadness", "😢"),
    1: ("Anger", "😠"),
    2: ("Love", "❤️"),
    3: ("Surprise", "😲"),
    4: ("Fear", "😨"),
    5: ("Joy", "😊")
}

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("🤖 EmotionSense AI")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 About")

st.sidebar.write(
    """
EmotionSense AI predicts human emotions
from text using NLP and Machine Learning.
"""
)

st.sidebar.subheader("🧠 Model")

st.sidebar.success("Multinomial Naive Bayes")

st.sidebar.subheader("📝 Vectorizer")

st.sidebar.info("CountVectorizer")

st.sidebar.subheader("⚙ NLP Pipeline")

st.sidebar.write("""
✅ Lowercase

✅ Remove Punctuation

✅ Remove Numbers

✅ Remove Emojis

✅ Tokenization

✅ Stopword Removal

✅ Count Vectorization
""")

st.sidebar.markdown("---")

st.sidebar.write("Built with ❤️ using Streamlit")


# ---------------- HEADER ---------------- #

col1, col2 = st.columns([1, 4])

with col1:
    st.image(logo, width=110)

with col2:
    st.markdown("""
    <h1 style="
    color:white;
    margin-top:10px;
    margin-bottom:0px;
    ">
    EmotionSense AI
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="
    color:white;
    font-size:18px;
    ">
    Detect Human Emotions from Text using NLP & Machine Learning
    </p>
    """, unsafe_allow_html=True)

# ---------------- MAIN CARD ---------------- #

st.markdown("<div class='main-card'>", unsafe_allow_html=True)

user_input = st.text_area(
    "Enter Text",
    placeholder="Example: I got promoted today!"
)

st.markdown("### 💡 Sample Inputs")

col1, col2 = st.columns(2)

with col1:
    st.code("I got promoted today!")
    st.code("I miss my childhood.")
    st.code("I love my family.")

with col2:
    st.code("I'm very angry.")
    st.code("I can't believe this!")
    st.code("I'm scared of tomorrow.")

predict = st.button("🚀 Predict Emotion")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ---------------- #

if predict:

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        vector = vectorizer.transform([user_input])

        prediction = model.predict(vector)[0]

        probabilities = model.predict_proba(vector)

        confidence = np.max(probabilities) * 100

        emotion, emoji = emotion_map[prediction]

        st.markdown(
        f"""
        <div class='result'>
            <h2>{emoji} Predicted Emotion</h2>
            <h1>{emotion}</h1>
            <h3>Confidence : {confidence:.2f}%</h3>
        </div>
        """,
        unsafe_allow_html=True
        )

        st.subheader("📊 Prediction Probability")

        labels = [
            emotion_map[i][0]
            for i in range(len(emotion_map))
        ]

        chart_data = {
            labels[i]: float(probabilities[0][i])
            for i in range(len(labels))
        }

        st.bar_chart(chart_data)

# ---------------- FOOTER ---------------- #

st.markdown("<hr>", unsafe_allow_html=True)

st.markdown(
"""
<div class='footer'>
Made with ❤️ by Sakshi

<br>

EmotionSense AI | Streamlit | Scikit-learn | NLTK
</div>
""",
unsafe_allow_html=True
)

