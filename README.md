# World Cup Match Predictor

A machine learning-based web application that predicts the outcome of football matches (Win / Loss / Draw) using historical match data and team performance metrics.

---

 ##Overview

This project combines data preprocessing, feature engineering, and machine learning to analyze past match results and generate predictions for future games. The application provides an interactive interface where users can select teams and instantly view predicted outcomes.

---

##  Key Features

- Interactive match prediction interface
- Real-time result generation
- Machine learning-based predictions
- Clean and user-friendly UI using Streamlit
- Modular and scalable code structure

---

##  How It Works

1. Historical match data is collected and cleaned  
2. Features like team strength, goals scored, and goals conceded are generated  
3. A Random Forest model is trained on this data  
4. The trained model is used to predict match outcomes  
5. Users interact with the model through a Streamlit web app  

---

##  Tech Stack

- **Programming Language:** Python  
- **Frontend:** Streamlit  
- **Machine Learning:** Scikit-learn  
- **Data Processing:** Pandas, NumPy  

---

##  Project Structure


world-cup-predictor/
│
├── app/
│ └── streamlit_app.py # Streamlit UI
│
├── src/
│ ├── data_preprocessing.py # Data cleaning
│ ├── feature_engineering.py # Feature creation
│ ├── train_model.py # Model training
│ └── predict.py # Prediction logic
│
├── data/
│ └── matches.csv # Dataset
│
├── models/
│ └── model.pkl # Trained model
│
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── .gitignore # Ignored files


---

##  Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/world-cup-predictor.git
cd world-cup-predictor

##2. Create Virtual Environment
python -m venv venv
Activate it:
venv\Scripts\activate

##3. Install Dependencies
pip install -r requirements.txt

##4. Run the Application
streamlit run app/streamlit_app.py

##Model Details
Algorithm: Random Forest Classifier
Input Features:
-Team strength
-Average goals scored
-Goals conceded
Output:
-Win / Loss / Draw prediction

##Limitations
-Predictions are based on historical data and may not reflect real-time team performance
-Model accuracy depends on data quality and feature selection
-Not intended for betting or real-world decision-making

##Future Improvements
-Improve prediction accuracy with advanced features
-Integrate live match data APIs
-Add team rankings and player statistics
-Deploy the application online
-Enhance UI with better visualizations


👨‍💻 Author
~Rian Mascarenhas


📄 License
This project is for educational purposes and is open for learning and experimentation.