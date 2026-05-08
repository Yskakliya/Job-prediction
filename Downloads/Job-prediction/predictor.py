import pandas as pd
import numpy as np
import sys
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.stem import PorterStemmer

TEST_SIZE = 0.2
ps = PorterStemmer()


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python project.py jobs.csv")

    df = load_data(sys.argv[1])
    df = preprocess(df)
    X, y, vectorizer = extract_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=42
    )

    models = train_models(X_train, y_train)
    models = tune_logistic_regression(models, X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    best_model_name = max(results, key=results.get)
    print(f"\nBest model: {best_model_name} ({results[best_model_name]:.4f})")

    plt.figure(figsize=(8, 4))
    plt.bar(
        results.keys(),
        results.values(),
        color=["gray", "steelblue", "seagreen", "tomato"]
    )
    plt.ylim(0, 1.0)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.show()

    with open("results.txt", "w") as f:
        for name, acc in results.items():
            f.write(f"{name}: {acc:.4f}\n")
        f.write(f"\nBest: {best_model_name} ({results[best_model_name]:.4f})\n")

    plot_learning_curves(models, X_train, y_train)

    rf = models["Random Forest"]
    importances = rf.feature_importances_
    feature_names = vectorizer.get_feature_names_out()
    indices = np.argsort(importances)[-10:]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title("Top 10 Important Features (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()

    demo_prediction(models, vectorizer)


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)

        if df.empty:
            sys.exit("Error: Dataset is empty.")

        for col in ["job_role", "skills"]:
            if col not in df.columns:
                sys.exit(f"Error: '{col}' column not found.")

        return df

    except FileNotFoundError:
        sys.exit("Error: File not found.")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9+#. ]", " ", text)
    words = [ps.stem(word) for word in text.split()]
    return " ".join(words)


def preprocess(df):
    required_cols = ["skills", "certifications"]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    df["text"] = (
        df["skills"].astype(str) + " " +
        df["certifications"].astype(str)
    )

    df["text"] = df["text"].apply(clean_text)

    return df


def extract_features(df):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(df["text"])
    y = df["job_role"]

    return X, y, vectorizer


def train_models(X_train, y_train):
    models = {
        "Baseline (Dummy)": DummyClassifier(
            strategy="most_frequent"
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        ),
        "Naive Bayes": MultinomialNB()
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

    return models


def tune_logistic_regression(models, X_train, y_train):
    print("\nTuning Logistic Regression with GridSearchCV...")

    params = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "saga"]
    }

    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        params,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best params:      {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

    models["Logistic Regression"] = grid_search.best_estimator_

    return models


def plot_learning_curves(models, X_train, y_train):
    real_models = {
        name: model for name, model in models.items()
        if name != "Baseline (Dummy)"
    }

    fig, axes = plt.subplots(1, len(real_models), figsize=(7 * len(real_models), 5))

    if len(real_models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, real_models.items()):
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            X_train,
            y_train,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 8)
        )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        ax.plot(train_sizes, train_mean, label="Train", color="steelblue")
        ax.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.15,
            color="steelblue"
        )

        ax.plot(train_sizes, val_mean, label="Validation", color="tomato")
        ax.fill_between(
            train_sizes,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.15,
            color="tomato"
        )

        ax.set_title(f"Learning Curve — {name}")
        ax.set_xlabel("Training size")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150)
    plt.show()


def evaluate_models(models, X_test, y_test):
    print("\n=== MODEL EVALUATION ===")

    accuracies = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc

        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))

        if name == "Baseline (Dummy)":
            continue

        top_classes = y_test.value_counts().head(30).index
        mask = y_test.isin(top_classes)

        y_test_small = y_test[mask]
        y_pred_series = pd.Series(y_pred, index=y_test.index)
        y_pred_small = y_pred_series[mask]

        cm = confusion_matrix(
            y_test_small,
            y_pred_small,
            labels=top_classes,
            normalize="true"
        )

        plt.figure(figsize=(16, 13))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=top_classes,
            yticklabels=top_classes
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix_{name.replace(' ', '_')}.png", dpi=150)
        plt.show()

    return accuracies


def demo_prediction(models_dict, vectorizer):
    print("\n=== DEMO PREDICTION ===")

    while True:
        user_input = input("\nEnter skills (or 'exit'): ")

        if user_input.lower() == "exit":
            break

        if not user_input.strip():
            continue

        cleaned_input = clean_text(user_input)
        input_tfidf = vectorizer.transform([cleaned_input])

        print(f"\n{'Model':<25} | {'Top-1 Prediction':<30} | {'Confidence'}")
        print("-" * 75)

        for name, model in models_dict.items():
            probs = model.predict_proba(input_tfidf)[0]
            classes = model.classes_

            best_idx = probs.argmax()
            role = classes[best_idx]
            conf = probs[best_idx] * 100

            print(f"{name:<25} | {role:<30} | {conf:.2f}%")

        for name, model in models_dict.items():
            print(f"\n--- Top-3 predictions: {name} ---")

            probs = model.predict_proba(input_tfidf)[0]
            classes = model.classes_

            top3 = sorted(
                zip(classes, probs),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            for i, (role, prob) in enumerate(top3):
                prefix = ">> 1." if i == 0 else f"   {i + 1}."
                print(f"{prefix}  {role:<30} - {prob * 100:.2f}%")

        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()