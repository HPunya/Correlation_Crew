# Correlation_Crew
Anvesham June Hackathon 2025

**Fake Job Detection System**

**Project Overview:** Fake job listings are a growing concern on job portals and social platforms. This project uses machine learning to detect whether a job posting is real or fake based on its text content. It includes a Streamlit web app that allows users to upload datasets, train/test the model, and get clear visual insights and fraud probability scores.

**Video Presentation:**
https://drive.google.com/file/d/1He2HI8DfMpU2razd-lVhvpxG5e9FAsU9/view?usp=sharing

**Key Features & Technologies Used**

**Features:**

- Upload training/test job listings as CSV
- Train a logistic regression model with text data
- Predict fraud risk with probability scores
- SHAP explainability plots
- WordCloud of suspicious job descriptions
- Single-job prediction form for manual input
- Visual charts (Pie, Histogram)

**Technologies:**
- Python
- Streamlit (Web UI)
- Pandas, NumPy (Data Handling)
- scikit-learn (ML model & TF-IDF)
- SHAP (Model explainability)
- Plotly, Matplotlib (Visualization)
- WordCloud

**Step by Step instructions:**

**1. Clone the Repository:**
git clone https://github.com/your-username/fake-job-detector.git
cd fake-job-detector

**2. Install Dependencies:**
Make sure you have Python 3.8+ installed, then run:
pip install -r requirements.txt

**3. Prepare Model Directory**
Create a models/ folder and add:
model.pkl – trained Logistic Regression model
vectorizer.pkl – fitted TF-IDF vectorizer

**4. Run the Application:**
Run the command: 
'streamlit run app.py' on the terminal

**5. Use the Web Interface**
Upload training data to train a model (or skip if already trained)
Upload test data to see predictions and visual insights
Use the form at the bottom to check any job manually

