# Sentiment Analysis (TF-IDF & ML)

This project performs sentiment analysis on Amazon Customer Reviews using Machine Learning (Logistic Regression) and TF-IDF vectorization. It predicts the sentiment score (1-5) and categorizes reviews into sentiment classes ranging from "Very Negative" to "Very Positive".

## Dataset

The project uses the **Amazon Customer Reviews** dataset.
- `Amazon Customer Reviews.csv`: The raw dataset containing customer reviews, scores, and other metadata.
- `Amazon Customer Reviews Cleaned Balanced.csv`: A processed version of the dataset (likely generated during the workflow) that handles class imbalance.

Key features used:
- `Text`: The content of the review.
- `Summary`: A brief summary of the review.
- `Score`: The rating given by the customer (1-5).
- `HelpfulnessNumerator` / `HelpfulnessDenominator`: Used to calculate a helpfulness ratio.

## Project Workflow

The Jupyter Notebook `Sentiment analysis (TF-IDF & ML).ipynb` covers the following steps:

1.  **Data Loading & Exploration**:
    -   Loading the dataset.
    -   Exploratory Data Analysis (EDA) to understand the distribution of scores and helpfulness.
    -   Feature Engineering: Creating `helpfulness_ratio`.

2.  **Data Preprocessing**:
    -   Text cleaning: Removing URLs, HTML tags, emojis, punctuation, and extra spaces.
    -   Handling missing values.

3.  **Feature Extraction**:
    -   **TF-IDF (Term Frequency-Inverse Document Frequency)** is used to convert text data into numerical vectors.

4.  **Model Training**:
    -   **Logistic Regression** is trained on the TF-IDF features.
    -   Class weights are computed and applied to handle the imbalance in review scores (since positive reviews often outnumber negative ones).

5.  **Evaluation**:
    -   The model is evaluated using accuracy, precision, recall, and F1-score.
    -   A confusion matrix is visualized to show the performance across different classes.

6.  **Prediction**:
    -   A `predict_review(text)` function is implemented to classify new, unseen reviews.

## Requirements

To run this notebook, you need the following Python libraries:

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Usage

1.  Ensure you have the dataset `Amazon Customer Reviews.csv` in the same directory.
2.  Open the notebook `Sentiment analysis (TF-IDF & ML).ipynb`.
3.  Run the cells sequentially to preprocess data, train the model, and view results.
4.  Use the `predict_review(text)` function at the end of the notebook to test custom text inputs.

## Results

The Logistic Regression model achieves an accuracy of approximately **70%** on the test set. It effectively identifies positive and negative sentiments, though distinguishing between adjacent ratings (e.g., 4 vs 5) can vary.
