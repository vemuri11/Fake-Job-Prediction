Fake Job Prediction Using Machine Learning and NLP
Author: vemuri11

Project Summary:
This project identifies whether a job posting is real or fake using machine learning and natural language processing (NLP). The goal is to help job seekers avoid fraudulent job advertisements by analyzing the text and patterns in job listings.

Key Features:
Uses real-world dataset from Kaggle with around 18,000 job listings.
Applies text preprocessing and feature extraction techniques such as TF-IDF.
Trains multiple machine learning models to detect fake job posts.
Provides a user-friendly graphical interface (GUI) using Python's Tkinter.
Displays model performance and comparison through visual graphs.

Dataset Details:
The dataset contains job postings with columns like:
Job title
Company profile
Job description
Requirements
Benefits
Location
Industry
And a target label: 'fraudulent' (0 = real, 1 = fake)

Workflow:
Dataset Upload
Users can upload the dataset directly through the GUI.
Data Preprocessing
Handles missing values.
Removes stopwords and punctuation.
Applies stemming and lemmatization using NLTK.
Text to Vector Conversion
Converts job description text to numerical features using TF-IDF vectorization.
Model Training and Evaluation

Supports multiple algorithms:
Support Vector Machine (SVM)
Random Forest
K-Nearest Neighbors (KNN)
Naive Bayes
Decision Tree
Multilayer Perceptron (MLP)
Measures accuracy, precision, recall, and F1-score for each model.
Displays a comparison graph for visual understanding.
Prediction Interface
Users can run any model directly from the GUI and view results in real-time.

How to Run the Project:
Step 1: Install required Python libraries
Step 2: Run the Python script called main.py
Step 3: Use the GUI to upload the dataset and click through each step:

Upload Dataset
Preprocess Dataset
Convert Text to TF-IDF
Run any of the algorithms
View comparison graph

Technologies Used:
Python
Scikit-learn (for ML models)
NLTK (for NLP processing)
Pandas and NumPy
Tkinter (for GUI)
Matplotlib (for graph visualization)

Why This Project Matters:
Fake job postings are a real and growing problem. This project demonstrates how data science can solve a real-world issue by combining data analysis, machine learning, and user interface design.

Possible Future Improvements:
Deploy the app as a web application (using Streamlit or Flask)
Add deep learning models like LSTM or BERT
Include real-time job scraping and live prediction
Improve UI with additional features like CSV export

Contact:
Author: vemuri11
GitHub: https://github.com/vemuri11/Fake-Job-Prediction.
