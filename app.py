"""
Medical Insurance Cost Prediction - Streamlit Demo
Run : streamlit run app.py
Needs insurance.csv in the same folder.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Insurance Cost Predictor", page_icon="🏥", layout="wide")

SEED = 42


# ---------------------------------------------------------------
# Data and model training (cached so it runs only once)
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv").drop_duplicates().reset_index(drop=True)


@st.cache_resource
def train_models():
    df = load_data()
    X = pd.get_dummies(df.drop(columns="charges"),
                       columns=["sex", "smoker", "region"], drop_first=True)
    y = df["charges"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

    models = {
        "Linear Regression": Pipeline([("s", StandardScaler()), ("m", LinearRegression())]),
        "Ridge Regression": Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "Lasso Regression": Pipeline([("s", StandardScaler()), ("m", Lasso(alpha=1.0))]),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=6,
                                               random_state=SEED),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.1,
                                                       max_depth=3, random_state=SEED),
    }

    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rows.append({"Model": name,
                     "MAE": round(mean_absolute_error(y_te, pred), 2),
                     "RMSE": round(float(np.sqrt(mean_squared_error(y_te, pred))), 2),
                     "R2": round(r2_score(y_te, pred), 4)})
        fitted[name] = model

    results = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    return df, X.columns.tolist(), fitted, results


def make_row(columns, age, sex, bmi, children, smoker, region):
    row = {"age": age, "bmi": bmi, "children": children,
           "sex_male": int(sex == "male"),
           "smoker_yes": int(smoker == "yes"),
           "region_northwest": int(region == "northwest"),
           "region_southeast": int(region == "southeast"),
           "region_southwest": int(region == "southwest")}
    return pd.DataFrame([row])[columns]


# ---------------------------------------------------------------
# Load
# ---------------------------------------------------------------
try:
    df, columns, fitted, results = train_models()
except FileNotFoundError:
    st.error("insurance.csv not found. Place it in the same folder as app.py.")
    st.stop()

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("🏥 Medical Insurance Cost Prediction")
st.caption("Using Machine Learning Regression Techniques")

# ---------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------
st.sidebar.header("Enter Person Details")
model_name = st.sidebar.selectbox("Model", results["Model"].tolist(), index=0)
age = st.sidebar.slider("Age", 18, 64, 30)
bmi = st.sidebar.slider("BMI", 15.0, 55.0, 25.0, 0.1)
children = st.sidebar.slider("Children", 0, 5, 1)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
smoker = st.sidebar.radio("Smoker", ["no", "yes"], horizontal=True)
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
predict = st.sidebar.button("Predict", type="primary", use_container_width=True)

model = fitted[model_name]

# ---------------------------------------------------------------
# Main area
# ---------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Prediction", "Model Comparison", "Data Insights"])

with tab1:
    if predict:
        x = make_row(columns, age, sex, bmi, children, smoker, region)
        cost = float(model.predict(x)[0])

        x_smoke = make_row(columns, age, sex, bmi, children, "yes", region)
        x_nosmoke = make_row(columns, age, sex, bmi, children, "no", region)
        cost_smoke = float(model.predict(x_smoke)[0])
        cost_nosmoke = float(model.predict(x_nosmoke)[0])

        st.subheader("Predicted Insurance Charges")
        st.metric(label=f"Estimated yearly cost ({model_name})", value=f"${cost:,.0f}")

        c1, c2, c3 = st.columns(3)
        c1.metric("As non-smoker", f"${cost_nosmoke:,.0f}")
        c2.metric("As smoker", f"${cost_smoke:,.0f}")
        c3.metric("Smoking adds", f"${cost_smoke - cost_nosmoke:,.0f}")

        avg = df["charges"].mean()
        st.info(f"Dataset average charge: ${avg:,.0f}. "
                f"This prediction is {abs(cost - avg) / avg * 100:.0f}% "
                f"{'above' if cost > avg else 'below'} average.")
    else:
        st.info("Set the details in the sidebar and click **Predict**.")

    st.subheader("Feature Importance (Random Forest)")
    rf = fitted["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=columns).sort_values(ascending=False)
    st.bar_chart(imp)

with tab2:
    st.subheader("Model Performance on Test Data")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.bar_chart(results.set_index("Model")["R2"])
    st.success(f"Best model: {results.loc[0, 'Model']} "
               f"(R² = {results.loc[0, 'R2']}, RMSE = {results.loc[0, 'RMSE']})")

with tab3:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Average charges: smoker vs non-smoker**")
        st.bar_chart(df.groupby("smoker")["charges"].mean())
    with c2:
        st.write("**Average charges by region**")
        st.bar_chart(df.groupby("region")["charges"].mean())

    st.write("**Age vs Charges**")
    st.scatter_chart(df, x="age", y="charges", color="smoker")
