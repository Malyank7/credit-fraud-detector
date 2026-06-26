import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="CCFRAUD",
    page_icon="🔍",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('../models/best_model.pkl')
    explainer = shap.TreeExplainer(model)
    return model, explainer

@st.cache_data
def load_test_data():
    X_test = np.load('../data/processed/X_test.npy')
    y_test = np.load('../data/processed/y_test.npy')
    feature_names = pd.read_csv(
        '../data/processed/feature_names.csv'
    ).iloc[:,0].tolist()
    return X_test, y_test, feature_names

model, explainer = load_model()
X_test, y_test, feature_names = load_test_data()
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# Header
st.title("🔍 Credit Card Fraud Detector")
st.markdown("Real-time fraud detection with SHAP explainability")
st.divider()

# Sidebar
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio(
    "Select Mode",
    ["Analyze Single Transaction", "Batch Analysis"]
)

# ── MODE 1: Single Transaction ──────────────────────────────
if mode == "Analyze Single Transaction":

    st.subheader("Select a Transaction")

    col1, col2 = st.columns(2)
    with col1:
        tx_type = st.selectbox(
            "Transaction type to analyze",
            ["Random Fraud Transaction", 
            "Random Legitimate Transaction",
            "Enter Transaction Index"]
        )
    with col2:
        if tx_type == "Enter Transaction Index":
            tx_idx = st.number_input(
                "Transaction index",
                min_value=0,
                max_value=len(X_test)-1,
                value=0
            )
        elif tx_type == "Random Fraud Transaction":
            fraud_indices = np.where(y_test == 1)[0]
            tx_idx = int(np.random.choice(fraud_indices))
            st.info(f"Selected fraud transaction index: {tx_idx}")
        else:
            legit_indices = np.where(y_test == 0)[0]
            tx_idx = int(np.random.choice(legit_indices))
            st.info(f"Selected legitimate transaction index: {tx_idx}")

    if st.button("🔍 Analyze Transaction", type="primary"):

        transaction = X_test_df.iloc[[tx_idx]]
        actual_label = int(y_test[tx_idx])

        # Predict
        pred = model.predict(transaction)[0]
        prob = model.predict_proba(transaction)[0][1]

        # Result display
        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Prediction",
                "🚨 FRAUD" if pred == 1 else "✅ LEGITIMATE"
            )
        with col2:
            st.metric(
                "Fraud Probability",
                f"{prob*100:.2f}%"
            )
        with col3:
            st.metric(
                "Actual Label",
                "🚨 FRAUD" if actual_label == 1 else "✅ LEGITIMATE"
            )

        # Correct/Wrong prediction
        if pred == actual_label:
            st.success("✓ Model predicted correctly")
        else:
            st.error("✗ Model predicted incorrectly")

        # SHAP Waterfall
        st.divider()
        st.subheader("🔎 Why did the model make this prediction?")

        shap_values = explainer.shap_values(transaction)
        sv = shap_values[1] if isinstance(
            shap_values, list) else shap_values
        base_value = explainer.expected_value[1] if isinstance(
            explainer.expected_value, list
        ) else explainer.expected_value

        explanation = shap.Explanation(
            values=sv[0],
            base_values=base_value,
            data=transaction.iloc[0].values,
            feature_names=feature_names
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(explanation, max_display=12, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.caption(
            "🔴 Red bars push prediction toward FRAUD | "
            "🔵 Blue bars push prediction toward LEGITIMATE"
        )

        # Top features table
        st.divider()
        st.subheader("📊 Top Contributing Features")
        feature_impact = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': sv[0],
            'Feature Value': transaction.iloc[0].values
        })
        feature_impact['Abs SHAP'] = feature_impact['SHAP Value'].abs()
        feature_impact = feature_impact.sort_values(
            'Abs SHAP', ascending=False
        ).head(10).reset_index(drop=True)
        feature_impact['Direction'] = feature_impact['SHAP Value'].apply(
            lambda x: '🔴 Toward Fraud' if x > 0 else '🔵 Toward Legit'
        )
        st.dataframe(
            feature_impact[['Feature', 'Feature Value', 
                            'SHAP Value', 'Direction']].round(4),
            use_container_width=True
        )

# ── MODE 2: Batch Analysis ───────────────────────────────────
else:
    st.subheader("📦 Batch Transaction Analysis")

    n_samples = st.slider(
        "Number of transactions to analyze", 
        20, 20000, 1000
    )

    if st.button("🔍 Run Batch Analysis", type="primary"):

        sample_idx = np.random.choice(
            len(X_test), n_samples, replace=False
        )
        X_sample = X_test_df.iloc[sample_idx]
        y_sample = y_test[sample_idx]

        preds = model.predict(X_sample)
        probs = model.predict_proba(X_sample)[:, 1]

        # Metrics
        from sklearn.metrics import f1_score, precision_score, recall_score
        f1 = f1_score(y_sample, preds)
        precision = precision_score(y_sample, preds)
        recall = recall_score(y_sample, preds)

        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Transactions Analyzed", n_samples)
        col2.metric("F1 Score", f"{f1:.4f}")
        col3.metric("Precision", f"{precision:.4f}")
        col4.metric("Recall", f"{recall:.4f}")

        # Fraud distribution
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Actual vs Predicted Fraud")
            summary = pd.DataFrame({
                'Category': ['Actual Fraud', 'Predicted Fraud',
                            'True Positives', 'False Positives',
                            'False Negatives'],
                'Count': [
                    int(y_sample.sum()),
                    int(preds.sum()),
                    int(((preds == 1) & (y_sample == 1)).sum()),
                    int(((preds == 1) & (y_sample == 0)).sum()),
                    int(((preds == 0) & (y_sample == 1)).sum())
                ]
            })
            st.dataframe(summary, use_container_width=True)

        with col2:
            st.subheader("Fraud Probability Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(probs[y_sample==0], bins=50, alpha=0.6,
                    color='steelblue', label='Legitimate')
            ax.hist(probs[y_sample==1], bins=50, alpha=0.6,
                    color='crimson', label='Fraud')
            ax.set_xlabel('Fraud Probability')
            ax.set_ylabel('Count')
            ax.legend()
            ax.set_title('Score Distribution')
            st.pyplot(fig)
            plt.close()

        # Global SHAP
        st.divider()
        st.subheader("🔎 Global Feature Importance (SHAP)")
        st.info("Calculating SHAP values for sample — may take 30 seconds...")

        shap_values = explainer.shap_values(X_sample)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(sv, X_sample, plot_type="bar",
                        max_display=12, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# Footer
st.divider()
st.caption(
    "Built with XGBoost + SHAP | "
    "Trained on Kaggle Credit Card Fraud Dataset | "
    "284,807 transactions | 0.17% fraud rate"
)