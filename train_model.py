import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def main():
    # Uses the dataset already present in this repo.
    # Expected columns in news.csv: title?, text, label
    df = pd.read_csv('news.csv')

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(
            "news.csv must contain columns: 'text' and 'label'. "
            f"Found columns: {list(df.columns)}"
        )

    X = df['text'].astype(str)
    y = df['label']

    # Vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_vectorized = vectorizer.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    # Train model
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    # Save model artifacts required by app.py
    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

    print("Model trained and saved successfully!")


if __name__ == '__main__':
    main()

