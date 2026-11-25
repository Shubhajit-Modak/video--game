# Spotify Song Popularity Prediction

## 1. Project Overview

Spotify Song Popularity Prediction is a supervised machine learning project that predicts the popularity category of a song using its audio features.

The original Spotify popularity score ranges from 0 to 100. For this project, the score is converted into three categories:

- 0–39 → Not Popular
- 40–69 → Moderately Popular
- 70–100 → Popular

The project uses nine Spotify audio features as the main predictors.

## 2. Objective

The main objectives of this project are:

1. Analyze Spotify song data.
2. Convert popularity scores into three classes.
3. Perform exploratory data analysis.
4. Train and compare three classification algorithms.
5. Select the best-performing model.
6. Analyze feature importance.
7. Predict the popularity category of a new song.
8. Provide a simple Streamlit web application.

## 3. Dataset

The dataset contains 114,000 songs and 21 columns.

The nine audio features used in the machine learning models are:

- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo

The original `popularity` column is used only to create the target category and is not used as an input feature.

## 4. Machine Learning Approach

### Target Variable

`popularity_category`

Classes:

| Popularity Score | Category |
|---|---|
| 0–39 | Not Popular |
| 40–69 | Moderately Popular |
| 70–100 | Popular |

### Train-Test Split

The dataset is divided into:

- 80% training data
- 20% testing data

Stratified sampling is used to maintain the class distribution.

### Models Used

Three supervised classification models are trained:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

StandardScaler is used for Logistic Regression. Tree-based models are trained without scaling.

## 5. Exploratory Data Analysis

The project includes:

- Popularity category distribution
- Correlation heatmap
- Audio feature distributions

These visualizations help understand the dataset and relationships between the selected audio features.

## 6. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Actual Test Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 37.71% | 0.5574 | 0.3771 | 0.4325 |
| Decision Tree | 69.55% | 0.7013 | 0.6955 | 0.6975 |
| Random Forest | 74.43% | 0.7491 | 0.7443 | 0.7456 |

## 7. Best Model

The Random Forest Classifier achieved the best overall performance.

### Random Forest Performance

- Accuracy: 74.43%
- Weighted Precision: 0.7491
- Weighted Recall: 0.7443
- Weighted F1-Score: 0.7456

Therefore, Random Forest is selected as the final model.

## 8. Feature Importance

The Random Forest feature importance results are:

| Rank | Feature | Importance |
|---|---|---:|
| 1 | Acousticness | 0.121976 |
| 2 | Loudness | 0.116570 |
| 3 | Energy | 0.114239 |
| 4 | Valence | 0.113539 |
| 5 | Speechiness | 0.111298 |
| 6 | Tempo | 0.110972 |
| 7 | Danceability | 0.110654 |
| 8 | Liveness | 0.108086 |
| 9 | Instrumentalness | 0.092666 |

These values represent the relative importance of the features within the trained Random Forest model.

## 9. Project Structure

```text
spotify-song-popularity/
│
├── data/
│   └── spotify_dataset.csv
│
├── models/
│   └── best_model.pkl
│
├── plots/
│   ├── popularity_distribution.png
│   ├── correlation_heatmap.png
│   ├── audio_feature_distributions.png
│   ├── logistic_confusion_matrix.png
│   ├── decision_tree_confusion_matrix.png
│   ├── random_forest_confusion_matrix.png
│   └── random_forest_feature_importance.png
│
├── train_model.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
