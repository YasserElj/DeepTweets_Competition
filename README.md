# DeepTweets Classification

The DeepTweets Classification project! This project was developed as part of a competition that involved classifying tweets into two categories: Politics and Sports. This README provides an overview of the project, its functionality, and the code used to achieve this classification.

## Table of Contents

- [Overview](#overview)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Submission](#submission)

## Overview

This project is centered around a tweet classification task where tweets are categorized into two classes: Politics and Sports. The goal is to correctly classify tweets into one of these two categories. The primary evaluation metric used in this competition is classification accuracy.

...

## Dependencies <!-- Add this section -->

- [Pandas](https://pandas.pydata.org/): Data manipulation and analysis library.
- [Numpy](https://numpy.org/): Scientific computing library for numerical operations.
- [Matplotlib](https://matplotlib.org/): Visualization library for creating plots and charts.
- [Seaborn](https://seaborn.pydata.org/): Data visualization library based on Matplotlib.
- [NLTK](https://www.nltk.org/): Natural Language Toolkit for text processing and analysis.
- [Scikit-learn](https://scikit-learn.org/): Machine learning library for training and evaluating models.

Make sure to install these dependencies before running the code. You can typically install them using the following command:

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn
```
## Data

The dataset used for this project consists of labeled tweets. The training dataset is loaded from `deeptweets/train.csv`, and the testing dataset is loaded from `deeptweets/test.csv`. The dataset includes the following columns:

- `TweetText`: The text of the tweet.
- `Label`: The label indicating whether the tweet belongs to Politics or Sports.

## Preprocessing

Data preprocessing is a crucial step in natural language processing tasks. In this project, the following preprocessing steps were performed:

1. **Label Encoding**: The labels were encoded to 0 for Sports and 1 for Politics.

2. **Text Cleaning**: Text data was cleaned using the following steps:
   - Removal of Twitter handles (e.g., @username).
   - Removal of special characters, numbers, and punctuation.
   - Removal of short words (words with a length less than 3 characters).
   - Removal of URLs.
   - Removal of single quotes.

3. **Text Tokenization and Stemming**: Tokenization involves breaking down text into individual words (tokens). The words were then stemmed using the Porter Stemmer, reducing words to their root forms to aid in feature extraction.

## Feature Extraction

Text data needs to be transformed into a format suitable for machine learning algorithms. In this project, the Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer was used for feature extraction. The TF-IDF vectorizer converts text data into numerical features that can be used to train machine learning models.

## Model Training

A Multinomial Naive Bayes classifier was used for this tweet classification task. The model was trained using the training data after feature extraction. The code for model training and evaluation is provided in the code section.

## Evaluation

The performance of the trained model is evaluated using classification accuracy. A confusion matrix is also plotted to visualize the model's performance.

## Submission

After training the model, it was used to make predictions on the test dataset. The predicted labels were converted to "Sports" or "Politics" based on a threshold, and the results were saved to a CSV file named `predictions.csv`.

Feel free to explore the code and adapt it for your own classification tasks. If you have any questions or need further assistance, please don't hesitate to reach out. Good luck with your tweet classification endeavors!
