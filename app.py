"""Streamlit interface for the customer churn Keras model."""

from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow import keras


MODEL_PATH = Path(__file__).with_name("churn_ann_model.keras")

# These are the min/max values fitted in MajorProjectFinal.ipynb.
TENURE_MAX = 72.0
MONTHLY_CHARGES_MIN = 18.25
MONTHLY_CHARGES_MAX = 118.75
TOTAL_CHARGES_MAX = 8684.80


@st.cache_resource(show_spinner="Loading churn model...")
def load_model():
    """Load the model once per Streamlit server process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    return keras.models.load_model(MODEL_PATH, compile=False)


def yes_no(label: str, *, default_yes: bool = False) -> int:
    """Return 1 for Yes and 0 for No from a compact radio control."""
    default_index = 0 if default_yes else 1
    return int(st.radio(label, ("Yes", "No"), index=default_index, horizontal=True) == "Yes")


def create_feature_vector(
    *,
    senior_citizen: int,
    partner: int,
    dependents: int,
    tenure: float,
    phone_service: int,
    multiple_lines: int,
    online_security: int,
    online_backup: int,
    device_protection: int,
    tech_support: int,
    streaming_tv: int,
    streaming_movies: int,
    paperless_billing: int,
    monthly_charges: float,
    total_charges: float,
    payment_method: str,
    internet_service: str,
    contract: str,
    gender: str,
) -> np.ndarray:
    """Create the 26 columns in exactly the model's training order."""
    return np.array(
        [[
            senior_citizen,
            partner,
            dependents,
            tenure / TENURE_MAX,
            phone_service,
            multiple_lines,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            paperless_billing,
            (monthly_charges - MONTHLY_CHARGES_MIN)
            / (MONTHLY_CHARGES_MAX - MONTHLY_CHARGES_MIN),
            total_charges / TOTAL_CHARGES_MAX,
            int(payment_method == "Bank transfer (automatic)"),
            int(payment_method == "Credit card (automatic)"),
            int(payment_method == "Electronic check"),
            int(payment_method == "Mailed check"),
            int(internet_service == "DSL"),
            int(internet_service == "Fiber optic"),
            int(internet_service == "No"),
            int(contract == "Month-to-month"),
            int(contract == "One year"),
            int(contract == "Two year"),
            int(gender == "Male"),
        ]],
        dtype=np.float32,
    )


st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉")
st.title("Customer Churn Prediction")
st.caption("Enter a customer's account details to estimate their likelihood of churn.")

try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the prediction model: {error}")
    st.stop()

with st.form("churn_form"):
    profile_col, account_col = st.columns(2)

    with profile_col:
        st.subheader("Customer profile")
        gender = st.selectbox("Gender", ("Female", "Male"))
        senior_citizen = int(st.checkbox("Senior citizen"))
        partner = yes_no("Has a partner")
        dependents = yes_no("Has dependents")
        tenure = st.number_input("Tenure (months)", min_value=0.0, max_value=72.0, value=12.0, step=1.0)

    with account_col:
        st.subheader("Account and charges")
        contract = st.selectbox("Contract", ("Month-to-month", "One year", "Two year"))
        payment_method = st.selectbox(
            "Payment method",
            (
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ),
        )
        paperless_billing = yes_no("Paperless billing", default_yes=True)
        monthly_charges = st.number_input(
            "Monthly charges ($)",
            min_value=MONTHLY_CHARGES_MIN,
            max_value=MONTHLY_CHARGES_MAX,
            value=70.0,
            step=0.01,
        )
        total_charges = st.number_input(
            "Total charges ($)",
            min_value=0.0,
            max_value=TOTAL_CHARGES_MAX,
            value=840.0,
            step=0.01,
        )

    st.subheader("Services")
    service_left, service_middle, service_right = st.columns(3)
    with service_left:
        phone_service = yes_no("Phone service", default_yes=True)
        multiple_lines = yes_no("Multiple lines")
        internet_service = st.selectbox("Internet service", ("DSL", "Fiber optic", "No"))
    with service_middle:
        online_security = yes_no("Online security")
        online_backup = yes_no("Online backup")
        device_protection = yes_no("Device protection")
    with service_right:
        tech_support = yes_no("Tech support")
        streaming_tv = yes_no("Streaming TV")
        streaming_movies = yes_no("Streaming movies")

    submitted = st.form_submit_button("Predict churn", type="primary")

if submitted:
    features = create_feature_vector(
        senior_citizen=senior_citizen,
        partner=partner,
        dependents=dependents,
        tenure=tenure,
        phone_service=phone_service,
        multiple_lines=multiple_lines,
        online_security=online_security,
        online_backup=online_backup,
        device_protection=device_protection,
        tech_support=tech_support,
        streaming_tv=streaming_tv,
        streaming_movies=streaming_movies,
        paperless_billing=paperless_billing,
        monthly_charges=monthly_charges,
        total_charges=total_charges,
        payment_method=payment_method,
        internet_service=internet_service,
        contract=contract,
        gender=gender,
    )
    churn_probability = float(model.predict(features, verbose=0)[0][0])

    st.metric("Churn probability", f"{churn_probability:.1%}")
    if churn_probability >= 0.5:
        st.error("This customer is predicted to churn.")
    else:
        st.success("This customer is predicted to stay.")
