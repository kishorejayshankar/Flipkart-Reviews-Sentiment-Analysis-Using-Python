# Flipkart Reviews Sentiment Analysis Uisng Python

This project aims to predict whether a review given on Flipkart is positive or negative using machine learning techniques. By analyzing user reviews and ratings, we can gain insights into product quality and provide recommendations for improvement.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Preprocessing](#preprocessing)
5. [Analysis](#analysis)
6. [Model Training](#model-training)
7. [Evaluation](#evaluation)
8. [Conclusion](#conclusion)

## Introduction
This project utilizes Python libraries such as Pandas, NLTK, Scikit-learn, Matplotlib, Seaborn, and Wordcloud for data preprocessing, analysis, and visualization. The main steps include importing the dataset, preprocessing the reviews, analyzing the data, converting text into vectors using TF-IDF, training a Decision Tree Classifier, and evaluating the model's performance.

## Installation
To run the code, you need to have Python installed on your system along with the required libraries. You can install the necessary libraries using pip:

```bash
pip install pandas scikit-learn nltk matplotlib seaborn wordcloud
```

Additionally, you need to download NLTK's stopwords by running the following Python code:

```python
import nltk
nltk.download('stopwords')
```

## Usage
1. Clone this repository to your local machine.
2. Download the dataset and place it in the project directory.
3. Run the provided Python script

## Preprocessing
The preprocessing step involves cleaning the reviews by removing punctuations, converting text to lowercase, and removing stopwords using NLTK. This ensures that the text data is ready for analysis and model training.

## Analysis
The analysis includes exploring unique ratings, visualizing rating distributions using countplots, and converting ratings into binary labels (positive or negative) based on a threshold (e.g., ratings <= 4 are considered negative).

## Model Training
The model training phase involves converting text data into vectors using TF-IDF (Term Frequency-Inverse Document Frequency) and splitting the dataset into training and testing sets. A Decision Tree Classifier is then trained on the training data.

## Evaluation
The model's performance is evaluated using accuracy score and confusion matrix. The confusion matrix provides insights into the model's predictions, including true positives, true negatives, false positives, and false negatives.

## Conclusion
The Decision Tree Classifier performs well for this dataset, achieving a high accuracy score. Future improvements may involve working with larger datasets and exploring other machine learning algorithms for better performance.
