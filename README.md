# Fake-Job-Detector

Overview
This repository contains an end-to-end data science project focused on the detection of fraudulent job postings. The project leverages Natural Language Processing (NLP) techniques and machine learning to build a robust predictive model. A key component of this project is the use of SHAP (SHapley Additive exPlanations) for model interpretability, providing transparent and explainable AI (XAI).

Technical Stack

Data Manipulation and Analysis: pandas, numpy

Data Visualization: matplotlib, seaborn, wordcloud

Machine Learning and NLP: scikit-learn

Model Explainability: shap

Model Persistence: joblib

Project Workflow

1. Exploratory Data Analysis (EDA)
Conducted a comprehensive EDA to understand the data's underlying structure and identify key features.

Performed data cleansing and preprocessing to handle missing values and create a unified text corpus by concatenating multiple text-based features into a single 'full_text' feature.

Generated word clouds to visualize the frequency of terms in both fraudulent and legitimate job postings, providing initial insights into discriminating vocabulary.

Utilized various plotting techniques to analyze the distribution of categorical variables such as employment type, required experience, and education level.

2. Feature Engineering and Vectorization
Engineered a 'full_text' feature that aggregates all textual information from the job postings, serving as the primary input for the model.

Implemented Term Frequency-Inverse Document Frequency (TF-IDF) vectorization to convert the raw text data into a high-dimensional feature space. The TF-IDF matrix represents the importance of each word in the corpus.

3. Predictive Modeling
Developed a Logistic Regression model, a powerful and interpretable linear model suitable for this binary classification task.

The dataset was partitioned into training and testing sets to ensure robust model evaluation and prevent overfitting.

The model was trained on the TF-IDF transformed training data.

Model performance was evaluated using a classification report, which includes precision, recall, and F1-score, providing a comprehensive view of the model's predictive power.

4. Model Interpretability with SHAP
To enhance model transparency, SHAP (SHapley Additive exPlanations) was employed.

A LinearExplainer was utilized to compute SHAP values for the features, quantifying the contribution of each word to the model's prediction for a given job posting.

Generated a SHAP summary plot to visualize the most influential features (words) and their impact on the likelihood of a job posting being fraudulent. This provides a global interpretation of the model's behavior.

5. Model Deployment
The trained Logistic Regression model and the TF-IDF vectorizer were serialized and saved as .pkl files using joblib. This allows for easy model persistence and deployment in a production environment.

Key Findings

The analysis and modeling revealed several key insights:

The Logistic Regression model achieved a high level of accuracy in distinguishing between fraudulent and legitimate job postings.

The SHAP analysis provided a clear understanding of which keywords and phrases are the strongest indicators of fraudulent activity, contributing to a more transparent and trustworthy model.

The project demonstrates a complete machine learning pipeline, from raw data to a deployable, interpretable model.
