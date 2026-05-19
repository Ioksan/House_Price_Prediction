# ============================================================
# UK HOUSE PRICE PREDICTION - STREAMLIT APP
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="UK House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ============================================================
# LOAD MODEL AND FEATURE OPTIONS
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("../models/best_house_price_model.pkl")

@st.cache_resource
def load_feature_options():
    return joblib.load("../models/feature_options.pkl")

model = load_model()
feature_options = load_feature_options()

# ============================================================
# HEADER
# ============================================================

st.title("🏠 UK House Price Prediction")
st.markdown("Predict UK house prices using a trained XGBoost machine learning model.")
st.warning("This prediction is for educational purposes only and should not be treated as a professional property valuation.")

st.divider()

# ============================================================
# LAYOUT - TWO COLUMNS
# ============================================================

col1, col2 = st.columns([1, 1])

# ============================================================
# LEFT COLUMN - INPUT FORM
# ============================================================

with col1:
    st.header("Property Details")

    input_data = {}
    all_filled = True

    for feature, options in feature_options.items():
        if feature == "min_year" or feature == "max_year":
            continue

        if isinstance(options, list):
            selected = st.selectbox(
                feature.replace("_", " ").title(),
                options=options,
                index=None,
                placeholder="Type or scroll to select...",
                key=f"select_{feature}"
            )

            if selected is not None:
                input_data[feature] = selected
            else:
                all_filled = False

        else:
            min_val = options["min"]
            max_val = options["max"]
            median_val = options["median"]

            if float(min_val).is_integer() and float(max_val).is_integer():
                input_data[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int(median_val),
                    step=1
                )
            else:
                input_data[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(median_val)
                )

    if not all_filled:
        st.info("Please select a value for all fields.")

    predict_button = st.button(
        "🔍 Predict House Price",
        use_container_width=True,
        disabled=not all_filled
    )

# ============================================================
# RIGHT COLUMN - RESULTS
# ============================================================

with col2:
    st.header("Prediction Results")

    if predict_button and all_filled:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        lower_bound = prediction * 0.85
        upper_bound = prediction * 1.15

        st.metric(
            label="Estimated House Price",
            value=f"£{prediction:,.0f}"
        )

        st.info(f"📊 Estimated range: **£{lower_bound:,.0f}** — **£{upper_bound:,.0f}**")

        st.divider()

        st.subheader("Property Summary")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            if "property_type" in input_data:
                type_map = {
                    "D": "Detached",
                    "S": "Semi-detached",
                    "T": "Terraced",
                    "F": "Flat",
                    "O": "Other"
                }
                st.metric("Property Type", type_map.get(input_data["property_type"], input_data["property_type"]))

            if "county" in input_data:
                st.metric("County", input_data["county"].title())

        with summary_col2:
            if "new_build" in input_data:
                st.metric("New Build", "Yes" if input_data["new_build"] == "Y" else "No")

            if "year" in input_data:
                st.metric("Year", input_data["year"])

        st.divider()

        st.subheader("Price Breakdown")

        breakdown_data = {
            "Category": ["Lower Estimate", "Predicted Price", "Upper Estimate"],
            "Price": [f"£{lower_bound:,.0f}", f"£{prediction:,.0f}", f"£{upper_bound:,.0f}"]
        }

        st.dataframe(
            pd.DataFrame(breakdown_data),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("👈 Fill in the property details on the left and click **Predict House Price**.")

        st.divider()

        st.subheader("About This Model")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        with metric_col1:
            st.metric("Algorithm", "XGBoost")
        with metric_col2:
            st.metric("RMSE", "£130,170")
        with metric_col3:
            st.metric("Training Data", "89,105 sales")

        st.markdown("""
        **How it works:**
        - Trained on real UK house price data (2015–2024)
        - Uses location, property type, and date features
        - Outliers removed for more accurate predictions
        - Evaluated using 5-fold cross-validation
        """)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("Data source: UK House Price Register (Kaggle) · Model: XGBoost · Built with Streamlit")