import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load dataset
data = pd.read_csv('preprocessed_data.csv')

data = data.dropna()

# Features and labels
X = data['text']  # The 'text' column is your feature
y = data['label']  # The 'label' column is your target

# Split the dataset into training and testing sets
X_train, y_train = X,y
# Convert the text data into numerical format using TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)  # You can change max_features based on your dataset size
X_train_tfidf = tfidf.fit_transform(X_train)

# Save the TF-IDF vectorizer
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

# Train the logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_tfidf, y_train)

# Save the trained model
joblib.dump(log_reg, 'logistic_regression_model.pkl')
