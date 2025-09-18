# 🚗 Old Car Price Classifier (Streamlit App)

This Streamlit app predicts whether a used car is **Expensive (>$20,000)** or **Affordable** based on its features.  
It uses a **Logistic Regression model** trained on the `used_car_price_dataset_extended.csv` dataset.

---

## 📌 Features

- Upload & explore the **Old Car dataset**
- Train a **Logistic Regression classifier**
- Interactive **sidebar inputs** for custom car features
- Predict whether a car is **Expensive** or **Affordable**
- Show **prediction probability**
- Display **model accuracy** and **confusion matrix**
- Dataset preview for transparency

---

## 🛠️ Installation

1. Clone the repository or copy the code file:

   ```bash
   git clone https://github.com/your-username/old-car-price-classifier.git
   cd old-car-price-classifier
Install the required dependencies:

bash
Copy
Edit
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
Make sure you have the dataset used_car_price_dataset_extended.csv in the project folder.

🚀 Running the App
Run the following command:

bash
Copy
Edit
streamlit run car_price_app.py
This will open the app in your browser at:

arduino
Copy
Edit
http://localhost:8501
📊 App Workflow
Load Data

Dataset: used_car_price_dataset_extended.csv

Creates binary target:

1 = Expensive (Price > $20,000)

0 = Affordable

Preprocessing

Selects numeric features

Standardizes features with StandardScaler

Model Training

Logistic Regression with max_iter=1000

Train-test split (80/20)

Prediction

Users enter car features in the sidebar

Model outputs prediction + probability

Evaluation

Accuracy score

Confusion Matrix visualization

📷 Example Screenshots
🏠 Main Dashboard

🎛️ Sidebar Inputs

📊 Confusion Matrix

📂 Project Structure
bash
Copy
Edit
old-car-price-classifier/
│── car_price_app.py              # Streamlit app
│── used_car_price_dataset_extended.csv   # Dataset
│── README.md                     # Documentation
│── docs/                         # Screenshots (optional)
🔮 Future Improvements
Add more ML models (Random Forest, Neural Network, KNN)

Allow users to upload their own dataset

Add feature importance visualization

Deploy on Streamlit Cloud

👨‍💻 Author
Mahendra Rathnayake
Data Analyst | Machine Learning Enthusiast

📫 Connect on LinkedIn

yaml
Copy
Edit

---

👉 Do you want me to also **include the dataset column descriptions** in the Markdown so users understand what each feature (like `mileage`, `car_age`, `fuel_type`) means?







You said:
yes


ChatGPT said:
Perfect 👍
Here’s the updated README.md with dataset column descriptions included:

markdown
Copy
Edit
# 🚗 Old Car Price Classifier (Streamlit App)

This Streamlit app predicts whether a used car is **Expensive (>$20,000)** or **Affordable** based on its features.  
It uses a **Logistic Regression model** trained on the `used_car_price_dataset_extended.csv` dataset.

---

## 📌 Features

- Upload & explore the **Old Car dataset**
- Train a **Logistic Regression classifier**
- Interactive **sidebar inputs** for custom car features
- Predict whether a car is **Expensive** or **Affordable**
- Show **prediction probability**
- Display **model accuracy** and **confusion matrix**
- Dataset preview for transparency

---

## 🗂️ Dataset Description

The dataset `used_car_price_dataset_extended.csv` contains the following columns:

| Column Name        | Type      | Description |
|--------------------|-----------|-------------|
| `price_usd`        | Numeric   | Price of the car in USD (Target variable) |
| `make_year`        | Numeric   | Manufacturing year of the car |
| `car_age`          | Numeric   | Car age (calculated as 2025 - make_year) |
| `mileage`          | Numeric   | Mileage of the car (in km or miles depending on dataset) |
| `engine_capacity`  | Numeric   | Engine size (e.g., in liters or cc) |
| `transmission`     | Categorical | Type of transmission (Manual / Automatic) |
| `fuel_type`        | Categorical | Fuel used (Petrol / Diesel / Hybrid / Electric) |
| `brand`            | Categorical | Car manufacturer (e.g., Toyota, BMW, Honda) |
| `color`            | Categorical | Color of the car |
| `insurance_valid`  | Categorical | Whether the insurance is valid (Yes/No) |
| `service_history`  | Categorical | Whether the car has a valid service history (Yes/No) |

📌 **Target Variable:**  
We create a binary column:
- `1 = Expensive` (Price > $20,000)  
- `0 = Affordable` (Price ≤ $20,000)  

---

## 🛠️ Installation

1. Clone the repository or copy the code file:

   ```bash
   git clone https://github.com/your-username/old-car-price-classifier.git
   cd old-car-price-classifier
Install the required dependencies:

bash
Copy
Edit
pip install streamlit pandas numpy scikit-learn matplotlib seaborn
Make sure you have the dataset used_car_price_dataset_extended.csv in the project folder.

🚀 Running the App
Run the following command:

bash
Copy
Edit
streamlit run car_price_app.py
This will open the app in your browser at:

arduino
Copy
Edit
http://localhost:8501
📊 App Workflow
Load Data

Dataset: used_car_price_dataset_extended.csv

Creates binary target: Expensive vs Affordable

Preprocessing

Selects numeric features

Standardizes features with StandardScaler

Model Training

Logistic Regression with max_iter=1000

Train-test split (80/20)

Prediction

Users enter car features in the sidebar

Model outputs prediction + probability

Evaluation

Accuracy score

Confusion Matrix visualization

📷 Example Screenshots
🏠 Main Dashboard

🎛️ Sidebar Inputs

📊 Confusion Matrix

📂 Project Structure
bash
Copy
Edit
old-car-price-classifier/
│── car_price_app.py                       # Streamlit app
│── used_car_price_dataset_extended.csv    # Dataset
│── README.md                              # Documentation
│── docs/                                  # Screenshots (optional)
🔮 Future Improvements
Add more ML models (Random Forest, Neural Network, KNN)

Allow users to upload their own dataset

Add feature importance visualization

Deploy on Streamlit Cloud

👨‍💻 Author
Mahendra Rathnayake
Data Analyst | Machine Learning Enthusiast

📫 Connect on LinkedIn