# train_model.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Load the dataset
data = pd.read_csv('Mail_Data.csv')

# Preprocess the data
data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data.Message, data.Spam, test_size=0.25, random_state=42)

# Create a pipeline with CountVectorizer and MultinomialNB
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

# Train the model
clf.fit(X_train, y_train)

# Save the trained model
with open('email_classifier_model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

# Print accuracy on test set
accuracy = clf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
