# Video Game Sales Prediction — GameSales AI

A complete Python + Streamlit machine-learning project based on the supplied `vgsales.csv` dataset and `Video_Game_Sales_Prediction.ipynb` workflow.

## Features
- Professional multi-page dashboard
- Dataset preview, search, schema and missing-value inspection
- Duplicate removal, numeric conversion and invalid-year handling
- Interactive EDA charts: sales distribution, genre, year and platform
- Linear Regression, Decision Tree Regressor and Random Forest Regressor
- MAE, MSE, RMSE and R² evaluation
- Five-fold cross-validation
- Optional Decision Tree and Random Forest grid-search tuning
- Low Sales / High Sales confusion matrices (threshold = 1 million)
- Best model saved with Joblib
- Interactive prediction form
- CSV exports of cleaned data and model results

## Dataset
The included dataset contains 16,598 records and these columns:
`Rank`, `Name`, `Platform`, `Year`, `Genre`, `Publisher`, `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, `Global_Sales`.

The model follows the notebook's feature selection:
- Inputs: `Platform`, `Genre`, `Year`, `Publisher`
- Target: `Global_Sales`

Sales values are in millions.

## Run on Windows
1. Install Python 3.10 or newer.
2. Extract the ZIP file.
3. Open the extracted `Video_Game_Sales_Prediction_Project` folder.
4. Double-click `run_app.bat`.

Or run manually in PowerShell:

```powershell
cd path\to\Video_Game_Sales_Prediction_Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser, usually at `http://localhost:8501`.

## How to use
1. Open **Overview** to see the dashboard.
2. Explore **Data Explorer**, **Data Cleaning**, and **Exploratory Analysis**.
3. Open **Model Training** and click **Train models**.
4. Optionally enable hyperparameter tuning (it takes longer).
5. Open **Predict Sales** after training to estimate sales for a game.
6. Model files and evaluation CSVs are saved in `artifacts/`.

## Cleaning note
The supplied notebook contains a year-filtering condition that sets plausible years (1950–2035) to missing. This app corrects that apparent condition error: years outside the plausible range are marked missing, and the pipeline imputes missing years using the training median.

## Methodology note
The notebook has a section titled confusion matrix while the main task is regression. To retain the requested operation, continuous actual/predicted sales are additionally grouped into Low Sales (<1 million) and High Sales (>=1 million), then displayed as confusion matrices. Regression metrics remain the primary model evaluation.

## Project structure

```text
Video_Game_Sales_Prediction_Project/
├── app.py
├── requirements.txt
├── run_app.bat
├── README.md
├── data/
│   └── vgsales.csv
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   └── modeling.py
└── artifacts/
    └── (generated after training)
```
