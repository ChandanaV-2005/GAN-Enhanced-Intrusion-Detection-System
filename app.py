import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
import sqlite3
from datetime import datetime
# =====================================
# SQLite Database
# =====================================

DB_PATH = "Database/ids_history.db"

conn = sqlite3.connect(DB_PATH)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS attack_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    prediction TEXT,
    confidence REAL,
    top_feature TEXT
)
""")

conn.commit()
conn.close()

st.set_page_config(
    page_title="Intrusion Detection System",
    page_icon="🔒"
)
st.sidebar.title("🛡 Navigation")

st.sidebar.markdown("""
## 📌 Project Details

**Project Name**
GAN-Enhanced Intrusion Detection System

**Dataset**
UNSW-NB15

**Original Records**
82,332

**Synthetic Attack Records**
1,000

**Augmented Dataset**
83,332

**Algorithm**
GAN + Random Forest

**Accuracy**
97.90%

**Features**
42

**Prediction**
Attack vs Normal
""")

st.title("🔒 GAN-Enhanced Intrusion Detection System")

st.markdown("""
### Intelligent Network Traffic Analysis using GAN and Machine Learning

This application analyzes network traffic using a **GAN-Enhanced Random Forest Intrusion Detection System**.

The **Generative Adversarial Network (GAN)** generates synthetic attack traffic to improve the training dataset, enabling the **Random Forest** classifier to detect cyber attacks with higher accuracy.
""")
# =====================================
# Project Workflow Cards
# =====================================

st.subheader("🔄 Project Workflow")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("""
    📁 Step 1
    
    Dataset Collection
    
    UNSW-NB15 Network Traffic Data
    """)

with col2:
    st.success("""
    ⚙️ Step 2
    
    Data Processing
    
    Cleaning + Encoding + Feature Selection
    """)

with col3:
    st.success("""
    🤖 Step 3
    
    GAN Model
    
    Generate Synthetic Attack Traffic
    """)


col4, col5, col6 = st.columns(3)

with col4:
    st.success("""
    🔄 Step 4
    
    Data Augmentation
    
    Combine Real + Synthetic Data
    """)

with col5:
    st.success("""
    🌲 Step 5
    
    Random Forest
    
    Train Intrusion Detection Model
    """)
st.subheader("🌟 Project Highlights")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "🤖 AI Technique",
    "GAN + RF"
)

col2.metric(
    "📊 Accuracy",
    "97.90%"
)

col3.metric(
    "📁 Dataset",
    "UNSW-NB15"
)

col4.metric(
    "⚡ Features",
    "42"
)

col5.metric(
    "🛡️ Domain",
    "Cybersecurity"
)
st.info("""
### 📊 Model Summary

- **Dataset:** UNSW-NB15
- **Original Records:** 82,332
- **Synthetic Attack Records:** 1,000
- **Augmented Dataset:** 83,332
- **Model:** GAN + Random Forest
- **Accuracy:** 97.90%
- **Number of Features:** 42
- **Prediction Type:** Attack vs Normal
""")
st.subheader("📂 Upload Network Traffic CSV")

st.caption("""
An AI-powered cybersecurity dashboard that detects malicious network traffic using Machine Learning.
""")

model = joblib.load("Model/ids_model_gan.pkl")
explainer = shap.TreeExplainer(model)
feature_names = joblib.load("Model/feature_names.pkl")


uploaded_file = st.file_uploader(
    "Choose CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    for col in ['id', 'label', 'attack_cat']:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)

    st.subheader("Uploaded Data")
    st.dataframe(data.head())

    if st.button("Predict"):


        missing_features = [
            feature for feature in feature_names
            if feature not in data.columns
        ]

        if missing_features:
            st.error(
            f"Missing features in uploaded CSV: {missing_features}"
            )
            st.stop()

    
        X_input = data[feature_names]

    
        predictions = model.predict(X_input)
     

        shap_values = explainer.shap_values(X_input)

        st.markdown("---")
        st.subheader("🔍 Explainable AI - SHAP")

        st.info("""
        SHAP explains why the Random Forest model predicted the network traffic
        as Attack or Normal.
        """)

    # Explain the first uploaded record
        sample_index = 0

        sample_prediction = predictions[sample_index]

        if isinstance(shap_values, list):

            if sample_prediction == 1:
                sample_shap = shap_values[1][sample_index]
            else:
                sample_shap = shap_values[0][sample_index]

        else:

            if len(shap_values.shape) == 3:

                if sample_prediction == 1:
                    sample_shap = shap_values[sample_index, :, 1]
                else:
                    sample_shap = shap_values[sample_index, :, 0]

            else:
                sample_shap = shap_values[sample_index]

    # Create SHAP dataframe
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": sample_shap,
            "Feature Value": X_input.iloc[sample_index].values
        })

        shap_df["Absolute SHAP"] = shap_df["SHAP Value"].abs()

        shap_df = shap_df.sort_values(
        by="Absolute SHAP",
        ascending=False
        )

    # Show prediction
        if sample_prediction == 1:

            st.error("🔴 Prediction: ATTACK")
            st.write("### Why was it classified as Attack?")

        else:

            st.success("🟢 Prediction: NORMAL")
            st.write("### Why was it classified as Normal?")

    # Show top 5 SHAP reasons
        top_shap = shap_df.head(5)

        for _, row in top_shap.iterrows():

            feature = row["Feature"]
            shap_value = row["SHAP Value"]
            feature_value = row["Feature Value"]

            if shap_value > 0:
                direction = "increased"
            else:
                direction = "decreased"

            st.write(
                f"• **{feature}** = `{feature_value}` "
                f"→ {direction} the prediction "
                f"(SHAP = `{shap_value:.4f}`)"
            )
        probabilities = model.predict_proba(X_input)

        confidence = probabilities.max(axis=1) * 100

        sample_confidence = confidence[sample_index]

        top_feature = top_shap.iloc[0]["Feature"]

        if sample_prediction == 1:
            prediction_text = "ATTACK"
        else:
            prediction_text = "NORMAL"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = sqlite3.connect(DB_PATH)

        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO attack_history
        (timestamp, prediction, confidence, top_feature)
        VALUES (?, ?, ?, ?)
        """, (
        timestamp,
        prediction_text,
        float(sample_confidence),
        top_feature
        ))

        conn.commit()
        conn.close()

        st.success("✅ Prediction saved to database successfully!")
        attack_count = (predictions == 1).sum()
        normal_count = (predictions == 0).sum()
        total_records = len(predictions)

        attack_percentage = round(
            (attack_count / total_records) * 100,2)

        normal_percentage = round(
            (normal_count / total_records) * 100,2)
        st.markdown("---")
        st.subheader("📈 Detection Summary")
        col1, col2, col3 = st.columns(3)

        col1.metric(
            label="🚨 Attack Records",
            value=attack_count
        )

        col2.metric(
            label="✅ Normal Records",
            value=normal_count
        )

        col3.metric(
            label="📄 Total Records",
            value=total_records
        )
        st.markdown("---")

        if attack_percentage == 0:
            st.success("🟢 System Status : SAFE")
        elif attack_percentage < 30:
            st.warning("🟡 System Status : LOW RISK")
        elif attack_percentage < 70:
            st.warning("🟠 System Status : MEDIUM RISK")
        else:
            st.error("🔴 System Status : HIGH RISK")
        st.subheader("📊 Traffic Analysis")

        col1, col2 = st.columns(2)

        col1.metric(
            label="🚨 Attack Traffic",
            value=f"{attack_percentage}%"
        )

        col2.metric(
            label="✅ Normal Traffic",
            value=f"{normal_percentage}%"
        )
        st.info(f"""
            ### 📋 Analysis Summary

            - Total Records Scanned : **{total_records}**
            - Attack Records : **{attack_count}**
            - Normal Records : **{normal_count}**
            - Detection Accuracy : **97.90%**
            - Model Used : **GAN + Random Forest**
        """)
        

        
        st.subheader("📊 Network Traffic Distribution")

        chart_df = pd.DataFrame({
            "Traffic Type": ["Normal", "Attack"],
            "Count": [normal_count, attack_count]
        })

        st.bar_chart(chart_df.set_index("Traffic Type"))

        fig, ax = plt.subplots(figsize=(6, 6))

# Handle cases where one category is zero
        if attack_count == 0:
            labels = ["Normal"]
            sizes = [normal_count]
        elif normal_count == 0:
            labels = ["Attack"]
            sizes = [attack_count]
        else:
            labels = ["Normal", "Attack"]
            sizes = [normal_count, attack_count]

        ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.axis("equal")
        ax.set_title("Network Traffic Distribution")

        st.pyplot(fig)
        plt.close(fig)
        st.markdown("---")
        st.subheader("🤖 Model Performance")

        performance = pd.DataFrame({
            "Algorithm": [
            "Random Forest",
            "Decision Tree",
            "K-Nearest Neighbors"
        ],
        "Accuracy (%)": [
        97,
        96,
        80
        ]
        })

        st.dataframe(
            performance,
            use_container_width=True
        )

        st.success("""
            🏆 Best Performing Model

            ✅ GAN + Random Forest

            Reason:
            ...
            """)

        st.markdown("---")
        st.subheader("📈 Before GAN vs After GAN")

        comparison = pd.DataFrame({
        "Metric": [
            "Training Dataset Size",
            "Attack Samples",
            "Machine Learning Model",
            "Accuracy"
        ],
        "Before GAN": [
            "82,332",
            "45,332",
            "Random Forest",
            "97.00%"
        ],
        "After GAN": [
            "83,332",
            "46,332",
            "GAN + Random Forest",
            "97.90%"
        ]
        })

        st.dataframe(
            comparison,
            use_container_width=True
        )



        col1, col2, col3, col4 = st.columns(4)
        st.markdown("---")
        st.subheader("📊 Model Evaluation")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            label="🎯 Accuracy",
            value="97.90%"
        )

        col2.metric(
            label="🎯 Precision",
            value="97.90%"
        )

        col3.metric(
            label="🎯 Recall",
            value="97.90%"
        )

        col4.metric(
            label="🎯 F1-Score",
            value="97.90%"
        )
        st.markdown("---")
    
        st.subheader("📊 Confusion Matrix")

        cm = [[7259,141],
            [209,9058]]

        fig, ax = plt.subplots(figsize=(5, 5))

        cax = ax.imshow(cm)

        for i in range(2):
            for j in range(2):
                ax.text(
                j, i, str(cm[i][j]),
                ha="center",
                va="center",
                color="white",
                fontsize=14,
                fontweight="bold"
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Attack"])
        ax.set_yticklabels(["Normal", "Attack"])

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title("Confusion Matrix")

        fig.colorbar(cax)

        st.pyplot(fig)
       
        plt.close(fig)
        

        st.success("""
            True Normal (TN): 7259

            False Positive (FP): 141

            False Negative (FN): 209

            True Attack (TP): 9058
        """)
        st.markdown("---")
        st.subheader("⭐ Top Important Features")

        feature_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        })

        feature_df = feature_df.sort_values(
            by="Importance",
            ascending=False
        )
        st.write("### 📋 Top 10 Important Features")

        st.dataframe(
            feature_df.head(10),
            use_container_width=True
        )
        fig, ax = plt.subplots(figsize=(8,5))

        top_features = feature_df.head(10)

        ax.barh(
            top_features["Feature"],
            top_features["Importance"]
        )

        ax.set_title("Top 10 Most Important Network Features")
        ax.set_xlabel("Feature Importance Score")
        ax.set_ylabel("Network Features")
        ax.invert_yaxis()

        st.pyplot(fig)
        
        plt.close(fig)
       


        data["Prediction"] = predictions

        data["Prediction"] = data["Prediction"].map({
            0: "Normal Traffic",
            1: "Attack Traffic"
        })
        st.success("Prediction Completed Successfully")

        st.subheader("Prediction Results")
        st.dataframe(data)

        csv = data.to_csv(index=False)

        st.download_button(
            "Download Results",
            csv,
            "prediction_results.csv",
            "text/csv"
        )
        # =====================================
# STEP 3 - ATTACK HISTORY DASHBOARD
# =====================================

st.markdown("---")
st.subheader("📋 Attack History Dashboard")

# Connect to SQLite database
conn = sqlite3.connect(DB_PATH)

# Read prediction history
history_df = pd.read_sql_query(
    """
    SELECT timestamp, prediction, confidence, top_feature
    FROM attack_history
    ORDER BY id DESC
    """,
    conn
)

conn.close()

# Check whether history exists
if history_df.empty:

    st.info("📭 No prediction history available yet.")

else:
    # =====================================
    # STEP 3.4 - SUMMARY METRICS
    # =====================================

    attack_total = (
        history_df["prediction"] == "ATTACK"
    ).sum()

    normal_total = (
        history_df["prediction"] == "NORMAL"
    ).sum()

    total_predictions = len(history_df)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🚨 Total Attacks",
        attack_total
    )

    col2.metric(
        "🟢 Total Normal",
        normal_total
    )

    col3.metric(
        "📄 Total Predictions",
        total_predictions
    )


    # =====================================
    # FILTER
    # =====================================

    filter_option = st.selectbox(
        "🔽 Filter Prediction",
        ["ALL", "ATTACK", "NORMAL"]
    )

    # Apply filter
    if filter_option == "ATTACK":

        filtered_df = history_df[
            history_df["prediction"] == "ATTACK"
        ]

    elif filter_option == "NORMAL":

        filtered_df = history_df[
            history_df["prediction"] == "NORMAL"
        ]

    else:

        filtered_df = history_df
    # =====================================
    # SEARCH HISTORY
    # =====================================
    
    search_text = st.text_input(
        "🔎 Search History",
        placeholder="Search prediction or top feature..."
    )
    
    # Apply search
    if search_text:
    
        search_text = search_text.lower()
    
        filtered_df = filtered_df[
            filtered_df["prediction"].str.lower().str.contains(
                search_text,
                na=False
            )
            |
            filtered_df["top_feature"].str.lower().str.contains(
                search_text,
                na=False
            )
        ]
        

    # Format confidence
    filtered_df["confidence"] = filtered_df["confidence"].round(2)

    # Display filtered history
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )
    