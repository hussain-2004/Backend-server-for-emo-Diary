# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import joblib

# # Load dataset
# data = pd.read_csv('DataSets/financial_emotions.csv')

# # Drop rows with missing values
# data = data.dropna()

# # Features and labels
# X = data['Sentence']  # Using only the 'content' column for TF-IDF
# y = data['Sentiment']  # The 'label' column is your target

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Convert the text data into numerical format using TfidfVectorizer
# tfidf = TfidfVectorizer(max_features=5000)  # You can change max_features based on your dataset size
# X_train_tfidf = tfidf.fit_transform(X_train)

# # Save the TF-IDF vectorizer
# joblib.dump(tfidf, 'tfidf_vectorizer/financial_emotions-tfidf_vectorizer.pkl')

# # Train the Random Forest model
# rf = RandomForestClassifier()  # You can set hyperparameters if needed
# rf.fit(X_train_tfidf, y_train)

# # Save the trained model
# joblib.dump(rf, 'Predictors/financial_emotions_predictor.pkl')

# # Optional: Evaluate the model on the test set
# X_test_tfidf = tfidf.transform(X_test)  # Transform the test set
# y_pred = rf.predict(X_test_tfidf)

# # Calculate and print evaluation metrics
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Precision:", precision_score(y_test, y_pred, average='weighted'))
# print("Recall:", recall_score(y_test, y_pred, average='weighted'))
# print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import joblib

# Load dataset
data = pd.read_csv('DataSets/financial_emotions.csv')

# Drop rows with missing values
data = data.dropna()

# Features and labels
X = data['Sentence']  # Using only the 'Sentence' column for BERT
y = data['Sentiment']  # The 'Sentiment' column is your target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Tokenize and encode sequences in the training set
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    list(y_train)
))

# Tokenize and encode sequences in the test set
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    list(y_test)
))

# Build the classification model
input_ids = tf.keras.Input(shape=(128,), dtype='int32', name='input_ids')
attention_mask = tf.keras.Input(shape=(128,), dtype='int32', name='attention_mask')

outputs = bert_model([input_ids, attention_mask])
cls_token = outputs.last_hidden_state[:, 0, :]

dense = tf.keras.layers.Dense(256, activation='relu')(cls_token)
dropout = tf.keras.layers.Dropout(0.3)(dense)
output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])

# Train the model
model.fit(train_dataset.shuffle(100).batch(16), epochs=3, batch_size=16)

# Evaluate the model on the test set
results = model.evaluate(test_dataset.batch(16))
print(f"Accuracy: {results[1]}")

# Save the BERT model and tokenizer
model.save('Predictors/financial_emotions_predictor2.h5')
tokenizer.save_pretrained('tfidf_vectorizer/financial_emotions-tfidf_vectorizer2.pkl')
