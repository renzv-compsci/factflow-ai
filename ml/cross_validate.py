from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
# Load Data
fake_df = pd.read_csv('ml/data/Fake.csv')
real_df = pd.read_csv('ml/data/True.csv')

# Add label: 0 for fake news, 1 for real news
fake_df['label'] = 0
real_df['label'] = 1

# Stacks two dataframes 
df = pd.concat([fake_df, real_df], ignore_index=True)

# Mix two data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check correct text column 
if 'text' in df.columns:
    X = df['text']
elif 'title' in df.columns:
    X = df['title']
else: 
    raise ValueError ("No 'text' or 'title' column")
# DataFrame df with columns 'text' 'label'
X = df['text']
y = df['label']

# Vectorize the text (turn text into numbers)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = vectorizer.fit_transform(X)

# Set up the model
model = LogisticRegression(max_iter=1000)

# Perform 5-fold cross - validation (cv = 5)
scores = cross_val_score(model, X_tfidf, y, cv=5, scoring='accuracy')

# Print results 
print("Cross-validation score for each fold: ",scores)
print("Mean Accuracy: ", scores.mean())
print("Standard deviation: ", scores.std())