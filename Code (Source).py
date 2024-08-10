# Importing necessary libraries
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download the IMDB dataset if not already present
!wget -nc https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# Extract the downloaded dataset
!tar -xf aclImdb_v1.tar.gz

# Function to load data from the IMDB dataset
def load_data(directory):
    data = []
    labels = []
    for label_type in ['pos', 'neg']:
        dir_name = os.path.join(directory, label_type)
        for fname in os.listdir(dir_name):
            if fname.endswith('.txt'):
                with open(os.path.join(dir_name, fname), encoding='utf-8') as f:
                    data.append(f.read())  # Append review text to data list
                labels.append(1 if label_type == 'pos' else 0)  # Append label (1 for positive, 0 for negative)
    return data, labels

# Load the training and test data
train_data, train_labels = load_data('aclImdb/train')
test_data, test_labels = load_data('aclImdb/test')

# Convert the data into pandas DataFrames
train_df = pd.DataFrame({'review': train_data, 'sentiment': train_labels})
test_df = pd.DataFrame({'review': test_data, 'sentiment': test_labels})

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_df['review'], train_df['sentiment'], test_size=0.2, random_state=42)

# Vectorize the text data using CountVectorizer, ignoring common English stop words
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)  # Fit and transform the training data
X_val_vec = vectorizer.transform(X_val)  # Transform the validation data

# Train a Naive Bayes model on the vectorized training data
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict and evaluate the model on the validation data
y_val_pred = model.predict(X_val_vec)
print(f'Validation Accuracy: {accuracy_score(y_val, y_val_pred)}')
print(classification_report(y_val, y_val_pred))

# Test the model on the test data
X_test_vec = vectorizer.transform(test_df['review'])  # Transform the test data
y_test_pred = model.predict(X_test_vec)  # Predict sentiments for the test data
print(f'Test Accuracy: {accuracy_score(test_df["sentiment"], y_test_pred)}')
print(classification_report(test_df['sentiment'], y_test_pred))

# Define a function to predict sentiment for new reviews
def predict_sentiment(review):
    review_vec = vectorizer.transform([review])  # Vectorize the review
    prediction = model.predict(review_vec)  # Predict sentiment
    sentiment = 'positive' if prediction[0] == 1 else 'negative'  # Convert prediction to sentiment label
    return sentiment

# Test the prediction function with some example reviews
reviews = [
    "I absolutely loved this movie, it was fantastic!",
    "This was the worst film I have ever seen.",
    "It was an okay movie, not too bad.",
    "The plot was dull and the acting was terrible."
]

# Print the predicted sentiment for each example review
for review in reviews:
    sentiment = predict_sentiment(review)
    print(f'Review: "{review}"')
    print(f'Sentiment: {sentiment}')
    print()

# Interactive user input to test the model with a new review
review = input("Enter a movie review: ")
sentiment = predict_sentiment(review)
print(f'\nSentiment of the review: {sentiment}')
