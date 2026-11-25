from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix

FEATURES = ["Platform", "Genre", "Year", "Publisher"]
TARGET = "Global_Sales"

def make_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric, ["Year"]),
        ("cat", categorical, ["Platform", "Genre", "Publisher"])
    ])

def train_models(df, test_size=0.2, tune=False):
    data = df.dropna(subset=[TARGET]).copy()
    X = data[FEATURES]
    y = data[TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    preprocessor = make_preprocessor()
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, min_samples_leaf=2),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=2
        )
    }
    pipelines = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe

    if tune:
        dt = Pipeline([("preprocessor", make_preprocessor()), ("model", DecisionTreeRegressor(random_state=42))])
        dt_grid = GridSearchCV(dt, {
            "model__max_depth": [10, 20, None],
            "model__min_samples_leaf": [1, 2]
        }, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
        dt_grid.fit(X_train, y_train)
        pipelines["Tuned Decision Tree"] = dt_grid.best_estimator_

        rf = Pipeline([("preprocessor", make_preprocessor()), ("model", RandomForestRegressor(random_state=42, n_jobs=-1))])
        rf_grid = GridSearchCV(rf, {
            "model__n_estimators": [80, 120],
            "model__max_depth": [20, None],
            "model__min_samples_leaf": [1, 2]
        }, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        pipelines["Tuned Random Forest"] = rf_grid.best_estimator_

    metrics, confusion = [], {}
    for name, model in pipelines.items():
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        metrics.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2 Score": r2_score(y_test, pred)
        })
        actual_class = np.where(y_test.to_numpy() < 1.0, "Low Sales", "High Sales")
        pred_class = np.where(pred < 1.0, "Low Sales", "High Sales")
        confusion[name] = confusion_matrix(
            actual_class, pred_class, labels=["Low Sales", "High Sales"]
        )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    # Evaluate the base three models with five-fold CV, matching the notebook approach.
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring={"MAE":"neg_mean_absolute_error", "MSE":"neg_mean_squared_error", "R2":"r2"},
            n_jobs=1
        )
        mae = -scores["test_MAE"]
        mse = -scores["test_MSE"]
        cv_rows.append({
            "Model": name,
            "Mean CV MAE": mae.mean(),
            "Std CV MAE": mae.std(ddof=1),
            "Mean CV MSE": mse.mean(),
            "Mean CV RMSE": np.sqrt(mse).mean(),
            "Mean CV R2": scores["test_R2"].mean()
        })
    cv_df = pd.DataFrame(cv_rows).sort_values("Mean CV MAE").reset_index(drop=True)
    best_base_name = cv_df.iloc[0]["Model"]
    best_name = best_base_name
    # Select from final test-set candidates using the cross-validation result for base models;
    # tuned candidates are compared on test MAE when tuning is enabled.
    metric_df = pd.DataFrame(metrics).sort_values("MAE").reset_index(drop=True)
    if tune:
        best_name = metric_df.iloc[0]["Model"]
    best_model = pipelines[best_name]

    root = Path(__file__).resolve().parents[1]
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    joblib.dump({"model": best_model, "model_name": best_name}, artifact_dir / "best_video_game_sales_model.pkl")
    metric_df.to_csv(artifact_dir / "final_model_comparison.csv", index=False)
    cv_df.to_csv(artifact_dir / "final_cross_validation_results.csv", index=False)
    pd.DataFrame([{"Model": k, "Confusion Matrix": str(v.tolist())} for k,v in confusion.items()]).to_csv(
        artifact_dir / "confusion_matrix_summary.csv", index=False
    )
    return {"metrics": metric_df, "cv": cv_df, "confusion": confusion, "best_model_name": best_name}

def load_saved_model(path):
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("model_name", "Saved model")
    return payload, "Saved model"

def predict_sales(model, platform, genre, year, publisher):
    row = pd.DataFrame([{
        "Platform": platform, "Genre": genre, "Year": year, "Publisher": publisher
    }])
    return max(0.0, float(model.predict(row)[0]))
