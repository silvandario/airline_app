(Written by ChatGPT)

# ✈️ Airline Satisfaction Prediction App

A powerful and interactive **Streamlit dashboard** to predict airline passenger satisfaction using **XGBoost** and **Random Forest**, with visual insights and segment analysis.

---

## 🚀 Features

- 🎯 **Predict customer satisfaction** based on 20+ service features
- 📊 **Feature importance charts** for both XGBoost and Random Forest
- 🧩 **Segment filtering** by travel class and trip type
- 📤 Upload your own CSV or
- ✍️ Manually test with customizable inputs

---

## 📦 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/silvandario/airline_app.git
cd airline_app
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install libomp for macOS (required for XGBoost)
```bash
brew install libomp
```

---

## ▶️ Run the App
```bash
.venv/bin/streamlit run app.py
```
Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 File Structure

```bash
├── app.py                     # Streamlit app
├── models/
│   ├── xgb_best_model.pkl     # Trained XGBoost model
│   ├── rf_best_model.pkl      # Trained Random Forest model
│   └── scaler.pkl             # Fitted RobustScaler
├── sample_data/
│   └── sample_clean.csv       # Optional CSV for default segment demo
├── requirements.txt           # All dependencies
├── .gitignore
└── README.md
```

---

## 🧠 Model Inputs

| Feature                      | Description                     |
|-----------------------------|---------------------------------|
| Age                         | Passenger age                   |
| Flight Distance             | Distance flown in km            |
| Delay                       | Total delay (departure + arrival) |
| Check-in Service, Cleanliness, etc. | Service ratings (1–5)      |
| One-hot Encoded Columns     | Gender, Class, Type of Travel   |

---

## 📈 Models Used

- **XGBoost**: Accurate, robust to noise, used for primary prediction
- **Random Forest**: Used for cross-validation and comparison

---

## 💡 Tip
Use the **manual input** tab to simulate customer satisfaction with different configurations and see how service factors influence model predictions.

---

> "Great data science is not just about models – it's about making decisions transparent and accessible."
