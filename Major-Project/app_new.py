import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

def format_inr(number):
    try:
        val = int(round(float(number)))
    except (ValueError, TypeError):
        return f"₹{number}"
    neg = val < 0
    s = str(abs(val))
    if len(s) <= 3:
        res = s
    else:
        last_three = s[-3:]
        other_parts = s[:-3]
        formatted = ""
        while len(other_parts) > 0:
            formatted = "," + other_parts[-2:] + formatted
            other_parts = other_parts[:-2]
        res = formatted.lstrip(",") + "," + last_three
    return f"-₹{res}" if neg else f"₹{res}"

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Used Car Market Analytics",
    page_icon="🚗",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("final_used_cars.csv")

# Convert StringDtype columns to object to avoid PyArrow compatibility errors in Streamlit
for col in df.columns:
    if isinstance(df[col].dtype, pd.StringDtype):
        df[col] = df[col].astype(object)

model = joblib.load("best_used_car_model.pkl")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/741/741407.png",
    width=120
)

st.sidebar.title("🚗 AI Used Car")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🚗 Price Prediction",
        "📊 Dashboard",
        "📈 Model Performance",
        "ℹ About"
    ]
)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "🏠 Home":

    st.title("🚗 AI Powered Used Car Market Analytics")

    st.markdown(
        """
Welcome to the **AI Used Car Market Analytics Platform**.

This application predicts the selling price of used vehicles using Machine Learning.

### ✨ Features

- 🚗 AI Price Prediction
- 📊 Interactive Dashboard
- 📈 Model Performance
- 📉 Market Analytics
- 🎯 User Friendly Interface

---
"""
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Cars",
        len(df)
    )

    col2.metric(
        "Brands",
        df["brand"].nunique()
    )

    col3.metric(
        "Average Price (INR)",
        format_inr(df['price'].mean() * 84.0)
    )

    col4.metric(
        "Maximum Price (INR)",
        format_inr(df['price'].max() * 84.0)
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    preview_df = df.head(10).copy()
    preview_df['price'] = preview_df['price'] * 84.0
    preview_df = preview_df.rename(columns={"price": "price (₹)"})
    st.dataframe(preview_df, use_container_width=True)

    st.markdown("---")

    st.subheader("Top 10 Brands")

    top_brand = (
        df["brand"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top_brand.columns = ["Brand", "Count"]

    fig = px.bar(
        top_brand,
        x="Brand",
        y="Count",
        color="Count",
        text_auto=True,
        title="Most Popular Brands"
    )

    st.plotly_chart(fig, use_container_width=True)
# --------------------------------------------------
# PRICE PREDICTION
# --------------------------------------------------

elif page == "🚗 Price Prediction":

    st.title("🚗 AI Used Car Price Prediction")

    st.write("Fill in the vehicle details below.")

    col1, col2 = st.columns(2)

    with col1:

        brand = st.selectbox(
            "Brand",
            sorted(df["brand"].unique())
        )

        model_name = st.selectbox(
            "Model",
            sorted(
                df[df["brand"] == brand]["model"].unique()
            )
        )

        # Helper function to get filtered options based on Brand & Model to avoid abnormalities
        def get_filtered_options(col):
            # Filter by Brand & Model
            filtered = df[(df["brand"] == brand) & (df["model"] == model_name)]
            opts = filtered[col].dropna().unique()
            if len(opts) == 0:
                # Fallback to Brand
                filtered = df[df["brand"] == brand]
                opts = filtered[col].dropna().unique()
            if len(opts) == 0:
                # Fallback to all
                opts = df[col].dropna().unique()
            return sorted(opts)

        fuel = st.selectbox(
            "Fuel Type",
            get_filtered_options("fuel_type")
        )

        engine = st.selectbox(
            "Engine",
            get_filtered_options("engine")
        )

        transmission = st.selectbox(
            "Transmission",
            get_filtered_options("transmission")
        )

    with col2:

        ext_col = st.selectbox(
            "Exterior Color",
            get_filtered_options("ext_col")
        )

        int_col = st.selectbox(
            "Interior Color",
            get_filtered_options("int_col")
        )

        accident = st.selectbox(
            "Accident History",
            get_filtered_options("accident")
        )

        clean_title = st.selectbox(
            "Clean Title",
            get_filtered_options("clean_title")
        )

        year = st.slider(
            "Model Year",
            2000,
            2026,
            2022
        )

        milage = st.number_input(
            "Mileage",
            min_value=0,
            value=25000
        )

    st.markdown("---")

    if st.button("🚀 Predict Price", use_container_width=True):

        car_age = 2026 - year

        input_df = pd.DataFrame({
            "brand": [brand],
            "model": [model_name],
            "milage": [float(milage)],
            "fuel_type": [fuel],
            "engine": [engine],
            "transmission": [transmission],
            "ext_col": [ext_col],
            "int_col": [int_col],
            "accident": [accident],
            "clean_title": [clean_title],
            "car_age": [int(car_age)]
        })

        X = df.drop(columns=["price"]).head(0)
        input_df = input_df[X.columns]

        prediction = model.predict(input_df)
        pred_price = max(0.0, prediction[0])
        pred_price_inr = pred_price * 84.0
        
        # Calculate acceptable demand range (+/- 7% negotiation margin)
        margin = 0.07
        lower_bound = pred_price_inr * (1 - margin)
        upper_bound = pred_price_inr * (1 + margin)
        
        val_est = format_inr(pred_price_inr)
        val_range = f"{format_inr(lower_bound)} - {format_inr(upper_bound)}"
        
        st.markdown(
            f"""
            <div style="background-color: #0f172a; padding: 30px; border-radius: 15px; border-left: 6px solid #10b981; margin-top: 25px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);">
                <h3 style="color: #94a3b8; margin: 0; font-size: 1rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">Estimated Market Value</h3>
                <h1 style="color: #10b981; margin: 10px 0; font-size: 3.5rem; font-weight: 800; line-height: 1;">{val_est}</h1>
                <hr style="border-color: #334155; margin: 20px 0;" />
                <h4 style="color: #cbd5e1; margin: 0 0 5px 0; font-size: 1.1rem; font-weight: 600;">Acceptable Demand Range (±7%)</h4>
                <p style="color: #38bdf8; font-size: 1.4rem; font-weight: 700; margin: 0;">{val_range}</p>
                <p style="color: #64748b; margin: 15px 0 0 0; font-size: 0.9rem; line-height: 1.5;">
                    🚗 Model: <strong>{brand} {model_name}</strong> ({year})<br/>
                    🛣 Mileage: <strong>{milage:,} mi</strong> | ⛽ Fuel: <strong>{fuel}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

elif page == "📊 Dashboard":

    st.title("📊 Used Car Market Dashboard")
    st.markdown("Explore key trends and distributions in the used car market dataset.")
    
    # 3 metrics on top
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Average Mileage", f"{int(df['milage'].mean()):,} mi")
    mcol2.metric("Newest Car Age", f"{int(df['car_age'].min())} years ({2026 - int(df['car_age'].min())})")
    mcol3.metric("Oldest Car Age", f"{int(df['car_age'].max())} years ({2026 - int(df['car_age'].max())})")
    
    st.markdown("---")
    
    # Layout with columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Price Distribution
        df_inr = df.copy()
        df_inr["price_inr"] = df_inr["price"] * 84.0
        fig_price = px.histogram(
            df_inr, 
            x="price_inr", 
            nbins=50, 
            title="Price Distribution (₹)",
            color_discrete_sequence=["#1f77b4"],
            labels={"price_inr": "Price (₹)"}
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Fuel Type Distribution
        fuel_counts = df["fuel_type"].value_counts().reset_index()
        fuel_counts.columns = ["Fuel Type", "Count"]
        fig_fuel = px.pie(
            fuel_counts, 
            values="Count", 
            names="Fuel Type", 
            title="Fuel Type Distribution",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_fuel, use_container_width=True)

    with col2:
        # Mileage vs Price
        sample_df = df.sample(min(1000, len(df)), random_state=42).copy()
        sample_df["price_inr"] = sample_df["price"] * 84.0
        fig_scatter = px.scatter(
            sample_df, 
            x="milage", 
            y="price_inr", 
            color="brand",
            title="Mileage vs Price (Sample of 1,000 Cars)",
            labels={"milage": "Mileage (mi)", "price_inr": "Price (₹)"},
            hover_data=["model", "car_age"]
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Transmission Distribution
        trans_counts = df["transmission"].value_counts().head(10).reset_index()
        trans_counts.columns = ["Transmission", "Count"]
        fig_trans = px.bar(
            trans_counts, 
            y="Transmission", 
            x="Count", 
            orientation="h",
            title="Top 10 Transmission Types",
            color="Count",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_trans, use_container_width=True)

# --------------------------------------------------
# MODEL PERFORMANCE PAGE
# --------------------------------------------------

elif page == "📈 Model Performance":

    st.title("📈 Model Performance & Evaluation")
    st.markdown("Compare machine learning models trained to predict used car prices.")
    
    # Display the model evaluation metrics in INR
    metrics_df = pd.DataFrame({
        "Model": ["Linear Regression", "Gradient Boosting", "Random Forest (Best)"],
        "MAE (₹)": [format_inr(25845.25 * 84), format_inr(22576.46 * 84), format_inr(18980.38 * 84)],
        "RMSE (₹)": [format_inr(137153.66 * 84), format_inr(135344.60 * 84), format_inr(134772.22 * 84)],
        "R² Score": [0.0797, 0.1038, 0.1114]
    })
    
    st.subheader("🏆 Model Comparison Metrics")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # R2 Comparison Chart
    fig_r2 = px.bar(
        metrics_df,
        x="Model",
        y="R² Score",
        color="Model",
        text_auto=".4f",
        title="Model R² Score Comparison (Higher is Better)",
        color_discrete_sequence=["#EF553B", "#00CC96", "#636EFA"]
    )
    st.plotly_chart(fig_r2, use_container_width=True)
    
    # MAE Comparison Chart in INR
    plot_mae_df = pd.DataFrame({
        "Model": ["Linear Regression", "Gradient Boosting", "Random Forest (Best)"],
        "MAE (₹)": [25845.25 * 84, 22576.46 * 84, 18980.38 * 84]
    })
    fig_mae = px.bar(
        plot_mae_df,
        x="Model",
        y="MAE (₹)",
        color="Model",
        text_auto="₹,.0f",
        title="Model Mean Absolute Error (MAE) Comparison (Lower is Better)",
        color_discrete_sequence=["#EF553B", "#00CC96", "#636EFA"],
        labels={"MAE (₹)": "Mean Absolute Error (₹)"}
    )
    st.plotly_chart(fig_mae, use_container_width=True)
    
    st.info(f"""
    **💡 Key Takeaways:**
    * **Random Forest** has the highest **R² Score (0.1114)** and lowest **MAE ({format_inr(18980.38 * 84)})**, making it our chosen deployment model.
    * The dataset contains significant variance in luxury vehicle prices, causing higher RMSE values across all models.
    """)

# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif page == "ℹ About":

    st.title("ℹ About the Project")
    st.markdown("""
    This project is an **AI-driven Used Car Market Analytics and Prediction Platform**. 
    
    ### ⚙ Tech Stack
    * **Frontend**: [Streamlit](https://streamlit.io/) for building high-performance web applications.
    * **Visualization**: [Plotly](https://plotly.com/) for interactive data analytics.
    * **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) pipelines for preprocessing and regressor ensembles.
    * **Data Storage**: [Pandas](https://pandas.pydata.org/) for data ingestion and manipulation.
    
    ### 📂 Dataset Information
    The model is trained on a comprehensive dataset of used vehicle postings containing variables such as brand, model, mileage, fuel type, engine configuration, transmission type, accident history, and title cleanliness.
    """)