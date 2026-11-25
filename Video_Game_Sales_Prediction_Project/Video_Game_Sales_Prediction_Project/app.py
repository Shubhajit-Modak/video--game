import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
from src.data_utils import load_data, clean_data, dataset_profile
from src.modeling import train_models, load_saved_model, predict_sales

st.set_page_config(page_title="GameSales AI", page_icon="🎮", layout="wide")
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "vgsales.csv"

st.markdown("""
<style>
.stApp {background: #0b1020; color: #edf2ff;}
[data-testid="stSidebar"] {background: #111a30;}
.block-container {padding-top: 1.8rem; max-width: 1500px;}
.hero {padding: 28px 32px; border-radius: 18px; background: linear-gradient(120deg,#172554,#312e81,#581c87); margin-bottom: 22px;}
.hero h1 {font-size: 2.35rem; margin: 0; color: white;}
.hero p {color: #dbeafe; font-size: 1rem; margin: 8px 0 0;}
.metric-card {background:#141f38;border:1px solid #293653;padding:16px;border-radius:14px;}
.small-note {color:#9fb0d0;font-size:.88rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_default_data():
    return load_data(DATA_PATH)

st.sidebar.title("🎮 GameSales AI")
st.sidebar.caption("Video Game Global Sales Prediction")
page = st.sidebar.radio("Navigation", ["Overview", "Data Explorer", "Data Cleaning", "Exploratory Analysis", "Model Training", "Predict Sales", "Project Guide"])
st.sidebar.divider()
st.sidebar.caption("Machine Learning • Regression • Model Evaluation")

uploaded = st.sidebar.file_uploader("Optional: use another CSV dataset", type=["csv"])
try:
    raw = pd.read_csv(uploaded) if uploaded else get_default_data()
except Exception as exc:
    st.error(f"Could not read dataset: {exc}")
    st.stop()

clean = clean_data(raw)
profile = dataset_profile(raw, clean)

if page == "Overview":
    st.markdown('<div class="hero"><h1>🎮 Video Game Sales Prediction</h1><p>Explore historical game sales, train machine learning models, compare performance and estimate worldwide sales.</p></div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total records", f"{len(raw):,}")
    c2.metric("Clean records", f"{len(clean):,}")
    c3.metric("Features", len(raw.columns))
    c4.metric("Missing cells", f"{int(raw.isna().sum().sum()):,}")
    st.subheader("Dataset snapshot")
    st.dataframe(raw.head(10), use_container_width=True, hide_index=True)
    left,right = st.columns(2)
    with left:
        genre = clean.groupby("Genre", as_index=False)["Global_Sales"].sum().sort_values("Global_Sales", ascending=False)
        fig = px.bar(genre, x="Genre", y="Global_Sales", title="Global sales by genre", color="Global_Sales", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        platform = clean.groupby("Platform", as_index=False)["Global_Sales"].sum().nlargest(12,"Global_Sales")
        fig = px.bar(platform, x="Platform", y="Global_Sales", title="Top 12 platforms", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    st.info("Start with Data Explorer and Data Cleaning, then train models from the Model Training page.")

elif page == "Data Explorer":
    st.title("📁 Data Explorer")
    st.write("Inspect the uploaded video game sales dataset.")
    a,b,c = st.columns(3)
    a.metric("Rows", f"{raw.shape[0]:,}")
    b.metric("Columns", raw.shape[1])
    c.metric("Duplicate rows", int(raw.duplicated().sum()))
    search = st.text_input("Search within the dataset")
    view = raw.copy()
    if search:
        mask = view.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        view = view[mask]
    st.dataframe(view, use_container_width=True, height=420)
    st.subheader("Column information")
    info = pd.DataFrame({"Column":raw.columns, "Data type":[str(x) for x in raw.dtypes], "Missing values":raw.isna().sum().values, "Missing %":(raw.isna().mean().values*100).round(2)})
    st.dataframe(info, use_container_width=True, hide_index=True)
    st.download_button("⬇ Download current dataset", raw.to_csv(index=False).encode("utf-8"), "vgsales_export.csv", "text/csv")

elif page == "Data Cleaning":
    st.title("🧹 Data Cleaning & Preprocessing")
    st.write("Cleaning follows the notebook workflow: remove duplicates, convert Year and Global_Sales to numeric, remove missing/negative target values and handle invalid years.")
    x,y = st.columns(2)
    x.metric("Original rows", f"{len(raw):,}")
    y.metric("Cleaned rows", f"{len(clean):,}")
    st.subheader("Missing values before cleaning")
    st.dataframe(raw.isna().sum().rename("Missing values").to_frame(), use_container_width=True)
    st.subheader("Cleaning summary")
    st.dataframe(profile["cleaning"], use_container_width=True, hide_index=True)
    st.subheader("Cleaned data preview")
    st.dataframe(clean.head(20), use_container_width=True)
    st.download_button("⬇ Download cleaned CSV", clean.to_csv(index=False).encode("utf-8"), "vgsales_cleaned.csv", "text/csv")

elif page == "Exploratory Analysis":
    st.title("📊 Exploratory Data Analysis")
    c1,c2 = st.columns(2)
    with c1:
        fig = px.histogram(clean, x="Global_Sales", nbins=55, title="Global sales distribution", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        bygenre = clean.groupby("Genre", as_index=False)["Global_Sales"].sum().sort_values("Global_Sales",ascending=False)
        fig = px.bar(bygenre, x="Genre", y="Global_Sales", title="Genre-wise global sales", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    year = clean.dropna(subset=["Year"]).groupby("Year", as_index=False)["Global_Sales"].sum()
    fig = px.line(year, x="Year", y="Global_Sales", markers=True, title="Year-wise global sales", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    platform = clean.groupby("Platform", as_index=False)["Global_Sales"].sum().nlargest(15,"Global_Sales")
    fig = px.bar(platform, x="Platform", y="Global_Sales", title="Top 15 platforms by global sales", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Sales values are expressed in millions of units, as represented in the source dataset.")

elif page == "Model Training":
    st.title("🧠 Train & Compare Machine Learning Models")
    st.write("Target: Global_Sales. Predictors: Platform, Genre, Year and Publisher.")
    st.warning("Training and cross-validation can take a few minutes depending on your computer.")
    test_size = st.slider("Test data percentage", 10, 35, 20, 5)
    run_tuning = st.checkbox("Run hyperparameter tuning (slower)", value=False)
    if st.button("🚀 Train models", type="primary", use_container_width=True):
        with st.spinner("Preparing data and training models..."):
            try:
                results = train_models(clean, test_size=test_size/100, tune=run_tuning)
                st.session_state["training_results"] = results
                st.success("Training completed. The best model has been saved in the artifacts folder.")
            except Exception as exc:
                st.exception(exc)
    result = st.session_state.get("training_results")
    if result:
        st.subheader("Test-set performance")
        st.dataframe(result["metrics"].round(4), use_container_width=True, hide_index=True)
        best = result["best_model_name"]
        st.success(f"Best model selected by lowest cross-validation MAE: {best}")
        st.subheader("Cross-validation results")
        st.dataframe(result["cv"].round(4), use_container_width=True, hide_index=True)
        st.subheader("Model MAE comparison")
        fig = px.bar(result["metrics"], x="Model", y="MAE", color="Model", template="plotly_dark", title="Lower MAE is better")
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Confusion matrices (sales category threshold: 1 million)")
        cols = st.columns(3)
        for col, (name, cm) in zip(cols, result["confusion"].items()):
            with col:
                st.markdown(f"**{name}**")
                fig = px.imshow(cm, text_auto=True, x=["Predicted Low","Predicted High"], y=["Actual Low","Actual High"], color_continuous_scale="Blues", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
        st.download_button("⬇ Download model metrics", result["metrics"].to_csv(index=False).encode(), "model_comparison.csv", "text/csv")
    else:
        st.info("No model results in this session yet. Press Train models to begin.")

elif page == "Predict Sales":
    st.title("🔮 Predict Global Video Game Sales")
    model_path = Path(__file__).parent / "artifacts" / "best_video_game_sales_model.pkl"
    if not model_path.exists():
        st.warning("Train the models first from the Model Training page. The trained model will then appear here.")
    else:
        model, model_name = load_saved_model(model_path)
        st.success(f"Loaded trained model: {model_name}")
        platforms = sorted(clean["Platform"].dropna().astype(str).unique())
        genres = sorted(clean["Genre"].dropna().astype(str).unique())
        publishers = sorted(clean["Publisher"].dropna().astype(str).unique())
        with st.form("prediction_form"):
            col1,col2 = st.columns(2)
            platform = col1.selectbox("Platform", platforms)
            genre = col2.selectbox("Genre", genres)
            year_min = int(clean["Year"].dropna().min())
            year_max = int(clean["Year"].dropna().max())
            year = col1.number_input("Release year", min_value=year_min, max_value=year_max, value=2020 if year_min <= 2020 <= year_max else year_max)
            publisher = col2.selectbox("Publisher", publishers)
            submitted = st.form_submit_button("Predict global sales", type="primary", use_container_width=True)
        if submitted:
            prediction = predict_sales(model, platform, genre, year, publisher)
            st.markdown("### Estimated global sales")
            st.metric("Predicted sales", f"{prediction:.3f} million")
            st.caption("This is a model estimate, not a guarantee of actual sales.")

else:
    st.title("ℹ️ Project Guide")
    st.markdown("""
    ### Project objective
    Predict worldwide video game sales using historical game metadata.

    ### Workflow
    1. Load the CSV dataset.
    2. Inspect data types, missing values and duplicate rows.
    3. Clean records and prepare categorical/numerical features.
    4. Split data into training and testing sets.
    5. Train Linear Regression, Decision Tree Regressor and Random Forest Regressor.
    6. Compare MAE, MSE, RMSE and R².
    7. Apply five-fold cross-validation and optional grid-search tuning.
    8. Convert continuous sales predictions into Low Sales / High Sales classes at a 1-million threshold for confusion-matrix analysis.
    9. Select and save the best model.
    10. Use the saved model to predict sales for a new game.

    ### Important note
    The notebook's Year-cleaning condition would replace valid years with missing values. This application corrects that apparent condition error by treating years outside 1950–2035 as invalid. The model uses the four predictors specified in the notebook: Platform, Genre, Year and Publisher.
    """)
