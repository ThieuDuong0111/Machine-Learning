import numpy as np
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords (từ không quan trọng) trong nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

data = {
    "Review": [
        "This product is amazing! I love it!",
        "Horrible experience, very disappointed.",
        "Best purchase I have ever made!",
        "Waste of money. Do not recommend.",
        "I like this product, very useful.",
        "Terrible quality, stopped working in 2 days.",
        "Very satisfied with my purchase.",
        "Not as good as expected, but okay.",
        "Absolutely fantastic! Worth every penny.",
        "Awful! I want a refund.", 
        "This is the worst product I have ever used.",
        "Absolutely terrible, do not waste your money!",
        "The quality is so bad, it broke after one use.",
        "I am extremely disappointed. Would never buy again!",
        "Horrible experience, nothing works as expected.",
        "I wish I could give this zero stars.",
        "The customer service is awful, they never respond.",
        "Totally useless, do not recommend at all.",
        "Not worth the price, complete scam.",
        "Very low quality, I regret this purchase."
    ],
    "Sentiment": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 1: Tích cực, 0: Tiêu cực
}

df = pd.DataFrame(data)

print(df.head())

corpus = []  # Danh sách chứa các câu đã tiền xử lý
stemmer = PorterStemmer()

for review in df["Review"]:
    # Loại bỏ dấu câu và chữ số
    review = re.sub(r'[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    
    # Loại bỏ stopwords (từ không quan trọng) và stemming (giảm từ về gốc)
    review = [stemmer.stem(word) for word in review if word not in stopwords.words('english')]
    
    # Ghép lại thành câu
    review = ' '.join(review)
    corpus.append(review)

print(corpus)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()  # Ma trận đặc trưng
y = df["Sentiment"].values  # Nhãn

# Chia tập train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kiểm tra kích thước tập dữ liệu
print("Training data size:", X_train.shape)
print("Testing data size:", X_test.shape)

model = MultinomialNB()
model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = model.predict(X_test)

print(y_pred)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {accuracy:.2f}")

new_review = ["This is so useless, this is too awful."]
new_review_cleaned = vectorizer.transform(new_review).toarray()
prediction = model.predict(new_review_cleaned)

print("Sentiment Prediction:", "Positive" if prediction[0] == 1 else "Negative")

