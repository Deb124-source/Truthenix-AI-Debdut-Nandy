# Truthenix

## AI Powered Fake News Detection System

Truthenix is a modern AI-powered Fake News Detection web application built using Flask, Machine Learning, and Natural Language Processing (NLP).

The system analyzes news articles and predicts whether the news is **REAL** or **FAKE** using a trained Machine Learning model.

---

# Preview

Truthenix features:

* Premium Glassmorphism UI
* Neon Gradient Branding
* AI-Powered News Analysis
* Real-Time Prediction System
* Responsive Modern Design

---

# Features

* Detects Fake and Real News
* Machine Learning Based Prediction
* NLP Text Processing
* TF-IDF Vectorization
* Logistic Regression Classifier
* Responsive User Interface
* Modern Dark Theme
* Fast Real-Time Analysis

---

# Tech Stack

## Frontend

* HTML5
* CSS3
* JavaScript

## Backend

* Flask
* Python

## Machine Learning

* Scikit-learn
* Logistic Regression
* TF-IDF Vectorizer
* Pandas
* NumPy

---

# Project Structure

```bash
Truthenix/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── model.pkl
├── vectorizer.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│── Fake_News_Detector_Debdut_Nandy.ipynb
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/Truthenix.git
```

---

## Move into Project Folder

```bash
cd Truthenix
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset

The project uses two datasets:

* Fake.csv
* True.csv

These datasets are used to train the Machine Learning model for fake news classification.

---

# Train the Model

Run:

```bash
python train_model.py
```

This will generate:

* model.pkl
* vectorizer.pkl

---

# Run the Application

```bash
python app.py
```

Then open:

```bash
http://127.0.0.1:5000
```

---

# How It Works

1. User pastes a news article
2. Text is cleaned and vectorized using TF-IDF
3. Logistic Regression model analyzes the news
4. System predicts:

   * REAL NEWS
   * FAKE NEWS

---

# Machine Learning Workflow

## Data Collection

Fake and real news datasets are collected.

## Data Preprocessing

* Text cleaning
* Stopword removal
* Vectorization

## Feature Extraction

TF-IDF Vectorizer converts text into numerical vectors.

## Model Training

A Logistic Regression classifier is trained on the processed dataset.

## Prediction

The trained model predicts whether the input news is fake or real.

---

# UI Highlights

* Premium Glassmorphism Design
* Smooth Modern Interface
* Neon Gradient Effects
* Dark AI Theme
* Responsive Layout
* Professional Startup Feel

---

# Future Improvements

* Deep Learning Integration
* Confidence Percentage Meter
* Live News API Integration
* News Source Credibility Checker
* Voice Input System
* AI Chat Assistant
* Multi-language Support
* User Authentication

---

# Requirements

```txt
flask
scikit-learn
pandas
numpy
```

---

# Screenshots


---

# Author

## Debdut Nandy

Machine Learning Enthusiast | Data Analyst | Web Developer

---

# License

This project is created for educational and portfolio purposes.

---

# Connect

If you like this project, feel free to star the repository and contribute.

⭐ Truthenix — Detecting Truth with AI
