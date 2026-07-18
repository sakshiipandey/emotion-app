# 🤖 EmotionSense AI

EmotionSense AI is a Machine Learning web application that predicts human emotions from text using Natural Language Processing (NLP). The application is built with Streamlit and uses a Multinomial Naive Bayes model trained on preprocessed text data.

## 🚀 Demo

https://emotion-app-349faj2ofmjwvvowximwna.streamlit.app/

## 📌 Features

* Predicts emotions from user-entered text
* Clean and interactive Streamlit interface
* Displays predicted emotion with emoji
* Shows prediction confidence
* Visualizes prediction probabilities
* Simple, fast, and easy to use

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Scikit-learn
* NLTK
* NumPy
* Joblib

---

## 🧠 Machine Learning Workflow

1. Text preprocessing

   * Lowercasing
   * Remove punctuation
   * Remove numbers
   * Remove emojis
   * Tokenization
   * Stopword removal

2. Feature Extraction

   * CountVectorizer

3. Model

   * Multinomial Naive Bayes

---

## 📂 Project Structure

```
emotion-app/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
├── assets/
└── screenshots/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/emotion-app.git
```

Go to the project directory:

```bash
cd emotion-app
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📸 Screenshots

Add screenshots inside the `screenshots` folder and include them here.

Example:

* Home Page
* Prediction Result
* Probability Chart

---

## 🎯 Future Improvements

* Deep Learning models (LSTM/BERT)
* Voice emotion detection
* Multi-language support
* Emotion analytics dashboard

---

## 👩‍💻 Author

**Sakshi**

If you like this project, consider giving it a ⭐ on GitHub.
