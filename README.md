# 💻 Laptop AI - Price Intelligence & Prediction Platform

An enterprise-grade machine learning and price comparison platform tailored for the Sri Lankan laptop market. Built using **Python**, **Scikit-Learn**, **Streamlit**, and **Plotly**.

---

## 🌟 Key Features

1. **AI Price Prediction (Sri Lanka Edition)**
   - Predicts fair market laptop prices in Sri Lankan Rupees (LKR / Rs.) using a trained **Random Forest Regressor** ($R^2 \approx 0.7748$).
   - Dynamically translates European benchmark valuations (EUR) into LKR using real-time foreign exchange sync or user-defined custom rates.
   - Evaluates 10 distinct hardware dimensions: Brand, Form Factor, Display Diagonal & Resolution, CPU, RAM, Storage Type & Capacity, GPU, OS, and Weight.

2. **Real-Time Retail Price Comparison**
   - Collects and aggregates actual listed prices from verified Sri Lankan retailers (including Laptop.lk, LuxuryX.lk, and dealer feeds).
   - Side-by-side comparison for identical laptop models across different local computer stores.
   - Price fluctuation tracking and historical pricing trend charts.
   - Comprehensive search and multi-criteria filtering by brand, CPU class, RAM, storage, seller, and price range.

3. **Empirical Dataset Analytics**
   - Interactive distributions, brand pricing premiums, and memory scaling benchmarks across 1,303 commercial laptop models.
   - Interactive Plotly visualizations with light theme aesthetics.

4. **Transparent Model Performance**
   - Plain-English breakdown of model evaluation metrics ($R^2$, MAE, RMSE) in both EUR and LKR.
   - Complete technical pipeline documentation.

---

## 🛠️ Technology Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
- **Machine Learning:** [Scikit-learn](https://scikit-learn.org/) (RandomForestRegressor Pipeline with OneHotEncoder)
- **Data Analysis & Visualization:** [Pandas](https://pandas.pydata.org/), [Plotly](https://plotly.com/python/)
- **Model Serialization:** [Joblib](https://joblib.readthedocs.io/)
- **Data Persistence:** SQLite3

---

## 📂 Project Structure

```text
laptop-price-predictor/
├── .streamlit/
│   └── config.toml                  # Streamlit UI theme configuration
├── data/
│   ├── laptop_comparison.db         # SQLite database storing retailer price offers
│   ├── laptop_price.csv             # Raw Kaggle laptop price dataset
│   └── sample_seller_feed.csv       # Sample Sri Lankan seller product feed
├── models/
│   └── laptop_price_model.joblib    # Trained Random Forest regression pipeline
├── outputs/
│   ├── laptop_cleaned.csv           # Cleaned dataset used by app & insights
│   ├── average_price_by_brand.png   # EDA visualization artifact
│   └── price_distribution.png       # EDA visualization artifact
├── price_comparison/
│   ├── __init__.py
│   ├── database.py                  # Database manager for offers & price history
│   ├── fetchers.py                  # Feed extractors for Sri Lankan computer retailers
│   └── service.py                   # High-level comparison service coordinator
├── app.py                           # Main Streamlit web application
├── eda.py                           # Exploratory data analysis script
├── train_model.py                   # Model training and evaluation script
├── requirements.txt                 # Python project dependencies
├── .gitignore                       # Git exclusion rules
└── README.md                        # Documentation
```

---

## 🚀 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/madhuharshaios/laptop-price-predictor.git
   cd laptop-price-predictor
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Deployment on Streamlit Community Cloud

1. Log into [Streamlit Community Cloud](https://share.streamlit.io/) with your GitHub account.
2. Click **Create app**.
3. Select this repository: `madhuharshaios/laptop-price-predictor`.
4. Set the branch to `main`.
5. Set the Main file path to `app.py`.
6. Click **Deploy!**
