# from flask import Flask, jsonify
# import joblib
# from flask_cors import CORS
# # from transformers import BertForSequenceClassification, BertTokenizer

# # Load the model and tokenizer from the directory
# # model = BertForSequenceClassification.from_pretrained("financial.zip")
# # tokenizer = BertTokenizer.from_pretrained("financial.zip")

# # Make predictions
# def predict_sentiment(text):
#     inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
#     outputs = model(**inputs)
#     predicted_label = torch.argmax(outputs.logits, dim=1).item()
#     return {0: 'neutral', 1: 'positive', 2: 'negative'}.get(predicted_label, "unknown")

# # Example prediction
# # print()

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Load the pre-trained models and TF-IDF vectorizer
# logistic_model = joblib.load('Predictors/logistic_regression_model.pkl')
# tweet_emotions_model = joblib.load('Predictors/tweet_emotions_predictor.pkl')
# tfidf_vectorizer = joblib.load('tfidf_vectorizer/tfidf_vectorizer.pkl')
# tweet_tfidf_vectorizer = joblib.load('tfidf_vectorizer/tweet_emotions-tfidf_vectorizer.pkl')


# financial_emotions_model = joblib.load('Predictors/financial_emotions_predictor.pkl')
# financial_tfidf_vectorizer = joblib.load('tfidf_vectorizer/financial_emotions-tfidf_vectorizer.pkl')

# mental_health_model = joblib.load('Predictors/mental_health_model.pkl')
# mental_health_tfidf_vectorizer = joblib.load('tfidf_vectorizer/mental_health_tfidf_vectorizer.pkl')

# toxicLang_tfidf_vectorizer = joblib.load('tfidf_vectorizer/toxic_lang.pkl')
# toxicLang_model = joblib.load('Predictors/toxic_lang.pkl')
# # Define prediction route with text in the URL
# @app.route('/predict/<path:mytext>', methods=['GET'])
# def predict(mytext):
#     # Transform the text using the loaded TF-IDF vectorizer for both models
#     transformed_text_logistic = tfidf_vectorizer.transform([mytext])
#     transformed_text_tweet_emotions = tweet_tfidf_vectorizer.transform([mytext])
#     transformed_text_financial_emotions = financial_tfidf_vectorizer.transform([mytext]) 
#     transformed_mental_health_emotions = mental_health_tfidf_vectorizer.transform([mytext]) 
#     transformed_toxic_lang = toxicLang_tfidf_vectorizer.transform([mytext]) 

#     # Make predictions
#     logistic_prediction = logistic_model.predict(transformed_text_logistic)
#     tweet_emotions_prediction = tweet_emotions_model.predict(transformed_text_tweet_emotions)
#     financial_emotions_prediction = financial_emotions_model.predict(transformed_text_financial_emotions)
#     # financial_emotion_prediction = predict_sentiment([mytext])
#     mental_health_prediction = mental_health_model.predict(transformed_text_financial_emotions)
#     toxic_lang_prediction = toxicLang_model.predict(transformed_toxic_lang)

#     # Return predictions directly as strings
#     return jsonify({
#         'emotion': int(logistic_prediction[0]),  # Return the class number directly
#         'tweet_emotion_prediction': str(tweet_emotions_prediction[0]),  # Return the string from the predictor
#         'financial_emotion_prediction':str(financial_emotions_prediction[0]),
#         'mental_health_prediction':str(mental_health_prediction[0]),
#         'toxic_language':str(toxic_lang_prediction[0])
#     })

# # Define home route for testing
# # @app.route('/')
# @app.route('/predict', methods=['POST'])
# def home():
#     return "ML Model API is up and running!"

# # Run the Flask app
# if __name__ == '__main__':
#     # app.run(debug=True)
#     app.run(port=5001)



from flask import Flask, jsonify, request
import joblib
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained models and TF-IDF vectorizer
logistic_model = joblib.load('Predictors/logistic_regression_model.pkl')
tweet_emotions_model = joblib.load('Predictors/tweet_emotions_predictor.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer/tfidf_vectorizer.pkl')
tweet_tfidf_vectorizer = joblib.load('tfidf_vectorizer/tweet_emotions-tfidf_vectorizer.pkl')
financial_emotions_model = joblib.load('Predictors/financial_emotions_predictor.pkl')
financial_tfidf_vectorizer = joblib.load('tfidf_vectorizer/financial_emotions-tfidf_vectorizer.pkl')
mental_health_model = joblib.load('Predictors/mental_health_model.pkl')
mental_health_tfidf_vectorizer = joblib.load('tfidf_vectorizer/mental_health_tfidf_vectorizer.pkl')
toxicLang_tfidf_vectorizer = joblib.load('tfidf_vectorizer/toxic_lang.pkl')
toxicLang_model = joblib.load('Predictors/toxic_lang.pkl')

# Define prediction route with text in the request body
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.json
        mytext = data.get('text', '')

        if not mytext:
            return jsonify({'error': 'No text provided'}), 400

        # Transform the text using the loaded TF-IDF vectorizer
        transformed_text_logistic = tfidf_vectorizer.transform([mytext])
        transformed_text_tweet_emotions = tweet_tfidf_vectorizer.transform([mytext])
        transformed_text_financial_emotions = financial_tfidf_vectorizer.transform([mytext]) 
        transformed_mental_health_emotions = mental_health_tfidf_vectorizer.transform([mytext]) 
        transformed_toxic_lang = toxicLang_tfidf_vectorizer.transform([mytext]) 

        # Make predictions
        logistic_prediction = logistic_model.predict(transformed_text_logistic)
        tweet_emotions_prediction = tweet_emotions_model.predict(transformed_text_tweet_emotions)
        financial_emotions_prediction = financial_emotions_model.predict(transformed_text_financial_emotions)
        mental_health_prediction = mental_health_model.predict(transformed_mental_health_emotions)
        toxic_lang_prediction = toxicLang_model.predict(transformed_toxic_lang)

        # Return predictions
        return jsonify({
            'emotion': int(logistic_prediction[0]),
            'tweet_emotion_prediction': str(tweet_emotions_prediction[0]),
            'financial_emotion_prediction': str(financial_emotions_prediction[0]),
            'mental_health_prediction': str(mental_health_prediction[0]),
            'toxic_language': str(toxic_lang_prediction[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define a health check route
@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'ML Model API is up and running!'}), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(port=5001)
