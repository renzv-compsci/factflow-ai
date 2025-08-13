# Core Training WorkFlow 

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

# Load datasets 
fake_df = pd.read_csv('ml/data/Fake.csv')
real_df = pd.read_csv('ml/data/True.csv')

# Add label: 0 for fake news, 1 for real news
fake_df['label'] = 0
real_df['label'] = 1

# Stacks two dataframes 
df = pd.concat([fake_df, real_df], ignore_index=True)

# Mix two data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split data to features (x) and labels (y)
X = df['text'] # News content 
y = df['label'] # Target label (0=fake, 1=real)

# Split data into training and test sets 
X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size = 0.2, random_state = 42
    # 0.2 20% of the data will be set aside for testing 
    # 80% will be used for training 
)

# Vectorize text using TF-IDF (turn text into numbers)
vectorizer = TfidfVectorizer(stop_words='english', max_df = 0.7) # 0.7 ignore words that appear more than 70% of the documents
X_train_tfidf = vectorizer.fit_transform(X_train) # Transforms text into a matrix of numbers 
X_test_tfidf = vectorizer.transform(X_test)

# Train a logistic regression 
model = LogisticRegression(max_iter = 1000) # Maximum number of iterations the algorithm will run
model.fit(X_train_tfidf, y_train) # Trains logistic regression using training data 

# Make predicitons on test set 
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .2f}\n")
print("Classification report: ")
print(classification_report(y_test, y_pred, target_names = ['Fake', 'Real']))