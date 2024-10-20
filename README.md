# iPhone Review Sentiment Analysis and Prediction

***Datasets doesn't include all iPhone models. Used available datasets at the time***

This project uses sentiment analysis and machine learning techniques to predict the reviews for the next unreleased iPhone model (iPhone 15). By analyzing previous iPhone reviews, we can build a model that classifies reviews as either positive or negative. The model can also provide insights into which words or phrases are commonly associated with positive and negative reviews, potentially predicting how users may react to the new iPhone release.

# Project Features
# 1. Data Preprocessing
Text cleaning removing special characters, lowercasing, tokenization, and stopwords.

Sentiment scores are calculated using the SentimentIntensityAnalyzer from the nltk library.

Reviews are categorized into positive or negative sentiment based on their compound sentiment score.

# 2. Model Used
Logistic Regression, Random Forest, and Naive Bayes models are trained on the cleaned and processed data.

The models use term frequency-inverse document frequency (TF-IDF) to convert textual data into numerical features.

# 3. Visualizations
Word clouds are generated to visually represent the most frequent terms in both positive and negative reviews of each iPhone model. 

# Files Included
Unbalanced and Unclean Datasets for each iPhone model:

***All datasets were found on Kaggle***
apple_iphone_11_reviews.csv - shows reviews of iPhone model 11

APPLE_iPhone_SE.csv - shows reviews of iPhone model SE

Apple-iPhone-7-32GB-Black.csv - shows reviews of iPhone model 7

iphone11pro-AmazonRenewed.csv - shows reviews of iPhone model 11 pro

iphone12-Apple.csv - shows reviews of Iphone model 12

iphone13-AmazonRenewed.csv" - shows reviews of iPhone model 13


# Installation & Requirements
To run this project, you may need the to install the following libraries :

nltk: For sentiment analysis using the SentimentIntensityAnalyzer.

pandas: For data manipulation and handling.

sklearn: For machine learning models (Logistic Regression, Random Forest, Naive Bayes) 

gensim: For word embedding (Word2Vec).

matplotlib & WordCloud: For data visualization

text2vec (if using alternative approaches): For tokenization and text vectorization in the R version
