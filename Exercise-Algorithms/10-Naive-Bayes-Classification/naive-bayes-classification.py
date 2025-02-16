import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = {
    'email': [
        "Free money now!!!", 
        "Hey, how are you?", 
        "Get a loan now", 
        "Limited offer, click here", 
        "Let's catch up tomorrow", 
        "Win a free iPhone today!", 
        "Project meeting at 3 PM"
    ],
    'label': ['spam', 'ham', 'spam', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)
print(df)

vectorizer = CountVectorizer()  # Chuyển đổi text thành ma trận số
X = vectorizer.fit_transform(df['email'])  # Biến đổi email thành ma trận đặc trưng
y = df['label']  # Nhãn (Spam hoặc Ham)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = MultinomialNB()  # Dùng Multinomial Naïve Bayes cho dữ liệu văn bản
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

new_email = ["You won a free vacation! Click here to claim"]
X_new = vectorizer.transform(new_email)  # Chuyển đổi email mới thành vector
prediction = model.predict(X_new)

print(f"Email: {new_email[0]} -> Prediction: {prediction[0]}")
