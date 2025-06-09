# Fake-News-Detection
# Fake News Detection using Logistic Regression

# Project Overview
This project builds a machine learning model to classify news articles as **fake** or **real** based on their text content. It uses natural language processing (NLP) techniques and Logistic Regression to predict the authenticity of news articles.

# Dataset
- The dataset contains labeled news articles categorized as fake or real.
- Data was split into 80% training and 20% testing sets with stratification.

# Approach
1. **Text Vectorization:** Converted news articles into numerical features using TF-IDF vectorization.
2. **Model Training:** Trained a Logistic Regression classifier on the training data.
3. **Evaluation:** Evaluated the model on test data using accuracy, precision, recall, F1-score, confusion matrix, and ROC AUC.
4. **Interpretation:** Analyzed top features influencing the model predictions.

# Results
- Achieved an accuracy of approximately 0.99 on the test set.
- ROC AUC score close to 1.0 indicates excellent class separation.
- Top predictive words for real news include: `reuters`, `wednesday`, `washington`, etc.
- Top predictive words for fake news include: `image`, `just`, `america`, `gop`, etc.

# How to Run
1. Clone the repo
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Run the Jupyter Notebook or Python script:
    ```
    jupyter notebook FakeNewsDetection.ipynb
    ```

# Dependencies
- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn

# Future Work
- Implement other classifiers (Random Forest, SVM).
- Use advanced NLP techniques like word embeddings or transformers.
- Deploy the model as a web app for real-time fake news detection.


