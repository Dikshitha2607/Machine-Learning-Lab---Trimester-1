import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("CarPrice_Assignment.csv")
    return df


@st.cache_resource
def train_regression_model(df):
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.drop(columns=["car_ID"], errors="ignore")

    X = df_clean.drop(columns=["price"])
    y = df_clean["price"]

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
    }

    return metrics


def make_car_visual(value, label, color):
    fig, ax = plt.subplots(figsize=(4.5, 2.1))
    body_width = 1.4 + (value / max(1, df["enginesize"].max())) * 1.1
    body_height = 0.55
    body_x = 0.18
    body_y = 0.38

    ax.add_patch(plt.Rectangle((body_x, body_y), body_width, body_height, facecolor=color, edgecolor="black", linewidth=2))
    ax.add_patch(plt.Rectangle((body_x + 0.22, body_y + 0.3), body_width * 0.5, body_height * 0.45, facecolor="lightgray", edgecolor="black", linewidth=1.5))
    ax.add_patch(plt.Circle((body_x + 0.22, body_y - 0.06), 0.12, color="black"))
    ax.add_patch(plt.Circle((body_x + body_width - 0.22, body_y - 0.06), 0.12, color="black"))
    ax.set_xlim(0, 2.3)
    ax.set_ylim(0, 1.5)
    ax.axis("off")

    ax.text(
        1.15,
        1.2,
        f"{label}: {value}",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    return fig


def price_impact_cards(df):
    med_engine = df["enginesize"].median()
    med_hp = df["horsepower"].median()
    med_weight = df["curbweight"].median()

    engine_low = df[df["enginesize"] <= med_engine]
    engine_high = df[df["enginesize"] > med_engine]
    hp_low = df[df["horsepower"] <= med_hp]
    hp_high = df[df["horsepower"] > med_hp]
    weight_low = df[df["curbweight"] <= med_weight]
    weight_high = df[df["curbweight"] > med_weight]

    metrics = {
        "Engine size": (
            engine_high["price"].mean() - engine_low["price"].mean(),
            f"${engine_high['price'].mean():,.0f} vs ${engine_low['price'].mean():,.0f}"
        ),
        "Horsepower": (
            hp_high["price"].mean() - hp_low["price"].mean(),
            f"${hp_high['price'].mean():,.0f} vs ${hp_low['price'].mean():,.0f}"
        ),
        "Curb weight": (
            weight_high["price"].mean() - weight_low["price"].mean(),
            f"${weight_high['price'].mean():,.0f} vs ${weight_low['price'].mean():,.0f}"
        ),
    }

    return metrics


def plot_relationship(df, x_col, y_col, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, color="#4c72b0")
    sns.regplot(data=df, x=x_col, y=y_col, scatter=False, color="#ff7f0e", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel("Price")
    return fig


df = load_data()
metrics = train_regression_model(df)
impact = price_impact_cards(df)


st.title("🚗 Car Price Prediction Dashboard")
st.caption("Multiple Linear Regression model for automobile price prediction")

with st.sidebar:
    st.header("Dataset Explore")
    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")
    st.write("Missing values:")
    st.write(df.isnull().sum().sum())

    st.header("Quick Filters")
    min_price, max_price = st.slider(
        "Price range",
        float(df["price"].min()),
        float(df["price"].max()),
        (float(df["price"].quantile(0.1)), float(df["price"].quantile(0.9))),
    )

    df_filtered = df[(df["price"] >= min_price) & (df["price"] <= max_price)]


tab1, tab2, tab3 = st.tabs(["Overview", "EDA", "Prediction"])

with tab1:
    st.subheader("Dataset Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.write("First 5 records")
        st.dataframe(df.head())
    with col2:
        st.write("Last 5 records")
        st.dataframe(df.tail())

    st.subheader("Data Summary")
    st.write(f"Shape: {df.shape}")
    st.write("Numerical features:")
    st.write(df.select_dtypes(include=np.number).columns.tolist())
    st.write("Categorical features:")
    st.write(df.select_dtypes(include="object").columns.tolist())
    st.write("Missing values by column:")
    st.dataframe(df.isnull().sum().to_frame(name="Missing").query("Missing > 0"))
    st.write("Duplicate rows:")
    st.write(df.duplicated().sum())
    st.write("Descriptive statistics for numerical attributes")
    st.dataframe(df.describe().T)

with tab2:
    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["price"], kde=True, color="#3a86ff", ax=ax)
    ax.set_title("Distribution of Car Prices")
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Interactive Price Impact Filters")
    engine_min, engine_max = float(df["enginesize"].min()), float(df["enginesize"].max())
    hp_min, hp_max = float(df["horsepower"].min()), float(df["horsepower"].max())
    weight_min, weight_max = float(df["curbweight"].min()), float(df["curbweight"].max())

    engine_threshold = st.slider("Engine Size Threshold", min_value=engine_min, max_value=engine_max, value=float(df["enginesize"].median()), step=5.0)
    hp_threshold = st.slider("Horsepower Threshold", min_value=hp_min, max_value=hp_max, value=float(df["horsepower"].median()), step=5.0)
    weight_threshold = st.slider("Curb Weight Threshold", min_value=weight_min, max_value=weight_max, value=float(df["curbweight"].median()), step=50.0)

    comp_col1, comp_col2, comp_col3 = st.columns(3)

    with comp_col1:
        engine_low = df[df["enginesize"] <= engine_threshold]
        engine_high = df[df["enginesize"] > engine_threshold]
        delta_engine = engine_high["price"].mean() - engine_low["price"].mean()
        st.metric(
            "Engine size price gap",
            f"+${delta_engine:,.0f}",
            f"Lower: ${engine_low['price'].mean():,.0f} | Higher: ${engine_high['price'].mean():,.0f}"
        )
        st.pyplot(make_car_visual(engine_threshold, "Selected engine size", "#4e79a7"))

    with comp_col2:
        hp_low = df[df["horsepower"] <= hp_threshold]
        hp_high = df[df["horsepower"] > hp_threshold]
        delta_hp = hp_high["price"].mean() - hp_low["price"].mean()
        st.metric(
            "Horsepower price gap",
            f"+${delta_hp:,.0f}",
            f"Lower: ${hp_low['price'].mean():,.0f} | Higher: ${hp_high['price'].mean():,.0f}"
        )
        st.pyplot(make_car_visual(hp_threshold, "Selected horsepower", "#59a14f"))

    with comp_col3:
        weight_low = df[df["curbweight"] <= weight_threshold]
        weight_high = df[df["curbweight"] > weight_threshold]
        delta_weight = weight_high["price"].mean() - weight_low["price"].mean()
        st.metric(
            "Curb weight price gap",
            f"+${delta_weight:,.0f}",
            f"Lower: ${weight_low['price'].mean():,.0f} | Higher: ${weight_high['price'].mean():,.0f}"
        )
        st.pyplot(make_car_visual(weight_threshold, "Selected curb weight", "#f28e2b"))

    st.write("Use the sliders to compare vehicles below and above each design threshold and see how the price usually increases as the car becomes stronger and heavier.")

    st.subheader("Price Relationship with Design Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Engine size impact",
            f"+${impact['Engine size'][0]:,.0f}",
            impact['Engine size'][1],
        )
        st.pyplot(make_car_visual(round(df["enginesize"].median(), 0), "Median engine size", "#4e79a7"))

    with col2:
        st.metric(
            "Horsepower impact",
            f"+${impact['Horsepower'][0]:,.0f}",
            impact['Horsepower'][1],
        )
        st.pyplot(make_car_visual(round(df["horsepower"].median(), 0), "Median horsepower", "#59a14f"))

    with col3:
        st.metric(
            "Curb weight impact",
            f"+${impact['Curb weight'][0]:,.0f}",
            impact['Curb weight'][1],
        )
        st.pyplot(make_car_visual(round(df["curbweight"].median(), 0), "Median weight", "#f28e2b"))

    st.subheader("Feature vs Price Scatterplots")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.pyplot(plot_relationship(df, "horsepower", "price", "Horsepower vs Price"))
    with col2:
        st.pyplot(plot_relationship(df, "enginesize", "price", "Engine Size vs Price"))
    with col3:
        st.pyplot(plot_relationship(df, "curbweight", "price", "Curb Weight vs Price"))

    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

    st.subheader("Top Correlations with Price")
    price_corr = corr["price"].drop("price").sort_values(ascending=False)
    st.dataframe(price_corr.to_frame(name="Correlation with Price"))

with tab3:
    st.subheader("Train and Evaluate Regression Model")
    st.write(f"MAE: ${metrics['MAE']:,.0f}")
    st.write(f"MSE: ${metrics['MSE']:,.0f}")
    st.write(f"RMSE: ${metrics['RMSE']:,.0f}")
    st.write(f"R² Score: {metrics['R2']:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(metrics["y_test"], metrics["y_pred"], alpha=0.7)
    min_val = min(metrics["y_test"].min(), metrics["y_pred"].min())
    max_val = max(metrics["y_test"].max(), metrics["y_pred"].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted Price")
    st.pyplot(fig)

    residuals = metrics["y_test"] - metrics["y_pred"]
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=metrics["y_pred"], y=residuals, ax=ax2, color="#ff7f0e")
    ax2.axhline(0, linestyle="--", color="black")
    ax2.set_xlabel("Predicted Price")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residual Plot")
    st.pyplot(fig2)

    st.subheader("Predict Price for a New Car")
    with st.form("new_car_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            enginesize = st.number_input("Engine Size", min_value=50, max_value=300, value=150)
            horsepower = st.number_input("Horsepower", min_value=40, max_value=300, value=110)
            curbweight = st.number_input("Curb Weight", min_value=1000, max_value=5000, value=2800)
            citympg = st.number_input("City MPG", min_value=10, max_value=60, value=24)
            highwaympg = st.number_input("Highway MPG", min_value=10, max_value=75, value=30)
        with col2:
            fueltype = st.selectbox("Fuel Type", df["fueltype"].unique().tolist())
            aspiration = st.selectbox("Aspiration", df["aspiration"].unique().tolist())
            carbody = st.selectbox("Body Style", df["carbody"].unique().tolist())
            drivewheel = st.selectbox("Drive Wheel", df["drivewheel"].unique().tolist())
            enginetype = st.selectbox("Engine Type", df["enginetype"].unique().tolist())
        with col3:
            symboling = st.number_input("Symboling", min_value=-2, max_value=4, value=1)
            doornumber = st.selectbox("Door Number", df["doornumber"].unique().tolist())
            enginelocation = st.selectbox("Engine Location", df["enginelocation"].unique().tolist())
            cylindernumber = st.selectbox("Cylinder Number", df["cylindernumber"].unique().tolist())
            fuelsystem = st.selectbox("Fuel System", df["fuelsystem"].unique().tolist())

        wheelbase = st.number_input("Wheelbase", min_value=80.0, max_value=130.0, value=95.0)
        carlength = st.number_input("Car Length", min_value=120.0, max_value=220.0, value=180.0)
        carwidth = st.number_input("Car Width", min_value=50.0, max_value=80.0, value=65.0)
        carheight = st.number_input("Car Height", min_value=40.0, max_value=80.0, value=54.0)
        boreratio = st.number_input("Bore Ratio", min_value=2.0, max_value=4.5, value=3.2)
        stroke = st.number_input("Stroke", min_value=2.0, max_value=4.5, value=3.2)
        compressionratio = st.number_input("Compression Ratio", min_value=6.0, max_value=12.0, value=9.0)
        peakrpm = st.number_input("Peak RPM", min_value=4000, max_value=7000, value=5200)

        submitted = st.form_submit_button("Predict Car Price")

    if submitted:
        sample = pd.DataFrame([
            {
                "symboling": symboling,
                "CarName": "custom_car",
                "fueltype": fueltype,
                "aspiration": aspiration,
                "doornumber": doornumber,
                "carbody": carbody,
                "drivewheel": drivewheel,
                "enginelocation": enginelocation,
                "wheelbase": wheelbase,
                "carlength": carlength,
                "carwidth": carwidth,
                "carheight": carheight,
                "curbweight": curbweight,
                "enginetype": enginetype,
                "cylindernumber": cylindernumber,
                "enginesize": enginesize,
                "fuelsystem": fuelsystem,
                "boreratio": boreratio,
                "stroke": stroke,
                "compressionratio": compressionratio,
                "horsepower": horsepower,
                "peakrpm": peakrpm,
                "citympg": citympg,
                "highwaympg": highwaympg,
            }
        ])

        for col in df.columns:
            if col not in sample.columns:
                sample[col] = np.nan

        sample = sample[df.columns.drop("price")]
        predicted_price = metrics["model"].predict(sample)[0]
        st.success(f"Estimated Selling Price: ${predicted_price:,.0f}")

        st.caption(
            "The prediction is based on the learned relationship between vehicle design, engine power, weight, and market price."
        )
