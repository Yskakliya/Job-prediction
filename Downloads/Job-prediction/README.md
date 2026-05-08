Skill-Based Job Role Predictor INF375 Final Project This application is an intelligent career assistant designed to bridge the gap between technical skills and job roles in the IT industry. Built with Python and the Flet framework, it provides a desktop and web interface where users can analyze their professional profile using machine learning.

Overview The project operates on a hybrid intelligence model. It primarily uses an ensemble of classic machine learning algorithms trained on real-world job data. For users who want deeper insights, it can optionally integrate with the Claude AI API to provide personalized career advice and identify missing skill sets.

Core Features Machine Learning Pipeline: The system utilizes Logistic Regression, Random Forest, and Naive Bayes classifiers. By using an ensemble approach, the app provides a "confidence" score based on how much the different models agree on a prediction.

Automated Text Processing: It includes a custom preprocessing engine that performs text cleaning and stemming to ensure that different variations of a skill (e.g., "Programming" vs "Programmer") are recognized correctly.

Intelligent Fallback: If no trained models are detected, the application remains functional by switching to a keyword-matching logic based on a predefined skill-role map.

Dynamic Training: Users can upload their own datasets in CSV format directly through the UI to retrain the models on the fly.

Technical Stack Frontend: Flet (Flutter-based UI for Python)

Machine Learning: Scikit-learn

Natural Language Processing: NLTK

Data Handling: Pandas

External Intelligence: Anthropic Claude API (Optional)

Installation and Usage Install dependencies: Run the following command to install the required libraries: pip install flet scikit-learn pandas nltk

Download NLTK resources: The application requires the 'punkt' tokenizer: python -c "import nltk; nltk.download('punkt')"

Run the application: For the desktop interface: python app.py

For the web interface: flet run --web app.py

Project Structure app.py: The main entry point containing the UI components and the prediction logic.

jobs.csv: The default dataset used for training the classifiers.

model_cache/: A directory created automatically to store serialized versions of your trained models and vectorizers.