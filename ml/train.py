# Core Training WorkFlow 
import joblib
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from utils import load_data, preprocess_dataframe, split_data

# Load and preprocess data
df = load_data('ml/data/training_set/Fake.csv', 'ml/data/training_set/True.csv')
df = preprocess_dataframe(df, text_column='text')

# Split data into train/test
X_train, X_test, y_train, y_test = split_data(df)

# Vectorize text using TF-IDF (turn text into numbers)
vectorizer = TfidfVectorizer(stop_words='english', max_df = 0.7) # 0.7 ignore words that appear more than 70% of the documents
X_train_tfidf = vectorizer.fit_transform(X_train) # Transforms text into a matrix of numbers 
X_test_tfidf = vectorizer.transform(X_test)

# Train a logistic regression 
model = LogisticRegression(max_iter = 1000) # Maximum number of iterations the algorithm will run
model.fit(X_train_tfidf, y_train) # Trains logistic regression using training data 

# Save model and vectorizer 
joblib.dump(model, 'logreg_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

# Evaluate the model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .2f}\n")
print("Classification report: ")
print(classification_report(y_test, y_pred, target_names = ['Fake', 'Real']))