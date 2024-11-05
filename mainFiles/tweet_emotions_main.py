import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load dataset
data = pd.read_csv('DataSets/tweet_emotions_preprocessed.csv')

# Drop rows with missing values
data = data.dropna()

# Features and labels
X = data['content']  # Using only the 'content' column for TF-IDF
y = data['sentiment']  # The 'label' column is your target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data into numerical format using TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)  # You can change max_features based on your dataset size
X_train_tfidf = tfidf.fit_transform(X_train)

# Save the TF-IDF vectorizer
joblib.dump(tfidf, 'tfidf_vectorizer/tweet_emotions-tfidf_vectorizer.pkl')

# Train the Logistic Regression model
lr = LogisticRegression(max_iter=1000)  # You can adjust max_iter if needed
lr.fit(X_train_tfidf, y_train)

# Save the trained model
joblib.dump(lr, 'Predictors/tweet_emotions_predictor.pkl')

# Optional: Evaluate the model on the test set
X_test_tfidf = tfidf.transform(X_test)  # Transform the test set
y_pred = lr.predict(X_test_tfidf)

# Calculate and print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
