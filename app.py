import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import plotly.express as px
import shap # type: ignore
from wordcloud import WordCloud # type: ignore

# Globals (to retain model + vectorizer)
model = None
vectorizer = None

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

st.set_page_config(page_title="Fake Job Detection System", layout="wide")
st.title("🕵️‍♂️ Fake Job Detection System")
st.markdown("Upload a job listings CSV and we'll detect possible frauds.")

# === 1. TRAINING PHASE ===
st.header("🧪 Step 1: Upload Training Data & Train the Model")
train_file = st.file_uploader("Upload training CSV", type=["csv"], key="train")

if train_file:
    df_train = pd.read_csv(train_file)

    st.subheader("Training Data Preview")
    st.dataframe(df_train.head())

    # Prepare training text data
    df_train["text"] = df_train["title"].fillna("") + " " + df_train["description"].fillna("")
    y_train = df_train["fraudulent"]

    # Train model
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(df_train["text"])

    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)

    st.success("✅ Model trained successfully!")

    # Show training F1-score
    train_preds = model.predict(X_train)
    train_f1 = f1_score(y_train, train_preds)
    st.metric("Training F1 Score", f"{train_f1:.4f}")

# === 2. TESTING PHASE ===
st.header("📦 Step 2: Upload Test Data & View Predictions")
test_file = st.file_uploader("Upload test CSV", type=["csv"], key="test")

if test_file and model and vectorizer:
    df_test = pd.read_csv(test_file)
    st.subheader("Test Data Preview")
    st.dataframe(df_test.head())

    # Preprocess test data
    df_test["text"] = df_test["title"].fillna("") + " " + df_test["description"].fillna("")
    X_test = vectorizer.transform(df_test["text"])

    # Predict
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    df_test["Predicted Label"] = preds
    df_test["Fraud Probability"] = probs

    st.subheader("🔍 Predictions")
    st.dataframe(df_test[["title", "Predicted Label", "Fraud Probability"]])

    # === Visuals ===
    st.subheader("📊 Fraud Probability Histogram")
    hist = px.histogram(df_test, x="Fraud Probability", nbins=30)
    st.plotly_chart(hist, use_container_width=True)

    pie_data = df_test["Predicted Label"].value_counts().rename({0: "Real", 1: "Fake"}).reset_index()
    # Count predicted labels
    label_counts = df_test["Predicted Label"].value_counts().reset_index()
    label_counts.columns = ["Predicted Label", "Count"]
    # Rename labels
    label_map = {0: "Real", 1: "Fake"}
    label_counts["Label"] = label_counts["Predicted Label"].map(label_map)
    # Plot pie chart
    pie_chart = px.pie(label_counts, names="Label", values="Count", title="Predicted Distribution")
    st.plotly_chart(pie_chart, use_container_width=True)

    st.subheader("🚨 Top 10 Most Suspicious Listings")
    st.table(df_test.sort_values("Fraud Probability", ascending=False)[["title", "Fraud Probability"]].head(10))

    # If true labels are present, show F1
    if "fraudulent" in df_test.columns:
        test_f1 = f1_score(df_test["fraudulent"], df_test["Predicted Label"])
        st.metric("Test F1 Score", f"{test_f1:.4f}")
        st.text("Classification Report")
        st.text(classification_report(df_test["fraudulent"], df_test["Predicted Label"]))
    
    #SHAP plot
    explainer = shap.Explainer(model, X_train)  # X_train should be your final processed training set
    shap_values = explainer(X_test)

    # Streamlit display
    st.subheader("SHAP Summary Plot")
    fig = shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(plt.gcf())

    #Word cloud
    text = " ".join(df_test[df_test['Predicted Label'] == 1]['description'].dropna())
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)

    st.subheader("Top Keywords in Fake Job Listings")
    st.image(wc.to_array())

elif test_file and not model:
    st.warning("⚠️ Please train the model first before uploading test data.")


# ✅ Real model-based preprocessing
def preprocess(df):
    # Combine 'title' and 'description' into one text column
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
    X = vectorizer.transform(df['text'])
    return X

# ✅ Use trained model to predict
def predict_jobs(df):
    X = preprocess(df)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]  # Probability of class 1 (fraud)

    df['prediction_label'] = preds
    df['prediction_label'] = df['prediction_label'].map({0: 'Real', 1: 'Fake'})
    df['fraud_probability'] = probs
    return df

# 📊 Plot fake vs real jobs
def plot_prediction_distribution(df):
    label_counts = df['prediction_label'].value_counts()

    fig, ax = plt.subplots()
    ax.bar(label_counts.index, label_counts.values, color=['crimson', 'royalblue'])
    ax.set_title("Fake vs Real Job Listings")
    ax.set_xlabel("Prediction Label")
    ax.set_ylabel("Number of Listings")
    st.pyplot(fig)

st.markdown("---")
st.header("🔍 Check a Single Job Posting")

# 👇 Streamlit form to take user input
with st.form("single_job_form"):
    title = st.text_input("Job Title")
    company = st.text_input("Company Name")
    location = st.text_input("Location")
    description = st.text_area("Job Description")
    requirements = st.text_area("Requirements")
    submit_button = st.form_submit_button("Check This Job")  # ✅ this creates submit_button!

# submitting the form
if submit_button:
    input_df = pd.DataFrame([{
        "title": title,
        "company_profile": company,
        "location": location,
        "description": description,
        "requirements": requirements
    }])

    try:
        # Preprocess using your vectorizer
        X_input = preprocess(input_df)
        pred = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0][1]

        st.success("✅ Prediction Complete!")
        st.markdown(f"🧠 **Prediction:** { 'Fake' if pred == 1 else 'Real' }")
        st.markdown(f"📊 **Fraud Probability:** {prob:.2%}")

        if prob > 0.75:
            st.warning("🚨 High Risk! This job may be a scam.")
        elif prob > 0.5:
            st.info("⚠ Moderate Risk. Review carefully before applying.")
        else:
            st.success("👍 Low Risk. Likely a genuine job post.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
