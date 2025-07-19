🕵️‍♂️ FAKE JOB PREDICTION USING MACHINE LEARNING AND NLP

📌 PROJECT SUMMARY
This project identifies whether a job posting is real or fake using machine learning and natural language processing (NLP). The goal is to help job seekers avoid fraudulent job advertisements by analyzing the text and patterns in job listings.

🌐 KEY FEATURES
Uses real-world dataset from Kaggle with around 18,000 job listings.
Applies text preprocessing and feature extraction techniques such as TF-IDF.
Trains multiple machine learning models to detect fake job posts.
Provides a user-friendly graphical interface (GUI) using Python's Tkinter.
Displays model performance and comparison through visual graphs.

📊 DATASET DETAILS
The dataset contains job postings with columns like:
Job title
Company profile
Job description
Requirements
Benefits
Location
Industry
And a target label: 'fraudulent' (0 = real, 1 = fake)

🔄 WORKFLOW
1. DATASET UPLOAD
Users can upload the dataset directly through the GUI.

2. DATA PREPROCESSING
Handles missing values.
Removes stopwords and punctuation.
Applies stemming and lemmatization using NLTK.

3. TEXT TO VECTOR CONVERSION
Converts job description text to numerical features using TF-IDF vectorization.

4. MODEL TRAINING AND EVALUATION
Supports multiple algorithms:
Support Vector Machine (SVM)
Random Forest
K-Nearest Neighbors (KNN)
Naive Bayes
Decision Tree
Multilayer Perceptron (MLP)
Measures accuracy, precision, recall, and F1-score for each model.
Displays a comparison graph for visual understanding.

5. PREDICTION INTERFACE
Users can run any model directly from the GUI and view results in real-time.

⚙️ HOW TO RUN THE PROJECT
Step 1: Install required Python libraries
Step 2: Run the Python script called main.py
Step 3: Use the GUI to upload the dataset and click through each step:
Upload Dataset
Preprocess Dataset
Convert Text to TF-IDF
Run any of the algorithms
View comparison graph

🧰 TECHNOLOGIES USED
Python
Scikit-learn (for ML models)
NLTK (for NLP processing)
Pandas and NumPy
Tkinter (for GUI)
Matplotlib (for graph visualization)

💡 WHY THIS PROJECT MATTERS
Fake job postings are a real and growing problem. This project demonstrates how data science can solve a real-world issue by combining data analysis, machine learning, and user interface design.

🔮 POSSIBLE FUTURE IMPROVEMENTS
Deploy the app as a web application (using Streamlit or Flask)
Add deep learning models like LSTM or BERT
Include real-time job scraping and live prediction
Improve UI with additional features like CSV export

📬 CONTACT
Author: vemuri11
GitHub: https://github.com/vemuri11/Fake-Job-Prediction
