# 🌙 SleepyWhisper

**Sleep Disorder Prediction Using Machine Learning**

> *A cozy, dark-themed web app that predicts sleep disorders using ML — because your sleep health matters* 💤

---

## ✨ What is SleepyWhisper?

SleepyWhisper is a full-stack web application that predicts whether you're **Healthy**, have **Insomnia**, or have **Sleep Apnea** — using simple health inputs that anyone can answer. No medical knowledge needed!

Built with 💜 using **Python Flask** and **Scikit-learn**.

---

## 🎯 Features

| Feature | Description |
|---------|------------|
| 🧠 **ML Prediction** | Predict sleep disorders using Gradient Boosting Classifier (97.23% accuracy!) |
| 📊 **Data Visualization** | 10 interactive charts analyzing 15,000 records |
| 📋 **Prediction History** | Track all your past predictions |
| 📄 **PDF Reports** | Download professional diagnostic reports |
| 🤖 **Sleep Chatbot** | AI assistant for sleep health tips |
| 💯 **Sleep Score** | Calculate your sleep health score (0-100) |
| 📈 **Trend Tracking** | Monitor how your sleep health improves over time |
| ⚡ **Model Comparison** | Side-by-side GBC vs QDA performance metrics |

---

## 🛠️ Tech Stack

```
Backend    → Python 3.12, Flask, Scikit-learn, Pandas, NumPy
Frontend   → HTML, CSS, JavaScript, Chart.js
ML Models  → Gradient Boosting Classifier, Quadratic Discriminant Analysis
Dataset    → Sleep Health & Lifestyle Dataset (15,000 records)
PDF        → FPDF2
Theme      → Dark mode with cozy rainy backgrounds 🌧️
```

---

## 📸 Screenshots

<img width="1916" height="917" alt="image" src="https://github.com/user-attachments/assets/216aafb8-7a27-4c84-a367-e1be2f7f8ce8" />
<img width="1872" height="902" alt="image" src="https://github.com/user-attachments/assets/0545ebc8-8906-4d58-aa51-c941d1e9cf95" />




---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/prachi109-pixel/SleepyWhisper.git
cd SleepyWhisper

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

# Install dependencies
pip install flask pandas numpy scikit-learn fpdf2

# Run the app
python app.py
```

Open your browser → `http://127.0.0.1:5000` 🌐

---

## 📊 Model Performance

| Metric | Gradient Boosting | QDA |
|--------|:-:|:-:|
| Training Accuracy | **98.41%** | 91.14% |
| Testing Accuracy | **97.23%** | 91.47% |
| Precision | **97.23%** | 91.59% |
| Recall | **97.23%** | 91.47% |
| F1-Score | **97.23%** | 91.48% |

---

## 📂 Project Structure

```
SleepyWhisper/
├── app.py                    # Flask backend
├── requirements.txt          # Dependencies
├── Sleep_Data_Sampled.csv    # Dataset
├── models/                   # Trained ML models (.pkl)
├── static/
│   └── images/               # Background images
├── templates/
│   ├── index.html            # Prediction page
│   ├── visualization.html    # Data visualization
│   ├── history.html          # Prediction history
│   ├── chatbot.html          # Sleep chatbot
│   ├── results.html          # Model comparison
│   ├── sleep_score.html      # Sleep score calculator
│   ├── trends.html           # Trend tracking
│   └── about.html            # About page
└── README.md
```

---

## 📝 Based On

IEEE Base Paper: *"Applying Machine Learning Algorithms for the Classification of Sleep Disorders"*
— Talal Sarheed Alshammari, IEEE Access, Volume 12, 2024

Our GBC achieved **97.23%** accuracy vs base paper's ANN at **92.92%** 🎉

---

## ⚠️ Disclaimer

This is a machine learning prediction tool, **not a medical diagnosis**. Please consult a healthcare professional for proper evaluation.

---


  🌙 <i>Sleep well, live well</i> 🌙

