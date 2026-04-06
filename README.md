# Loan Eligibility Predictor - Flask Application

A web-based machine learning application for predicting loan eligibility using Flask. The application allows users to train models, make single predictions, and process batch predictions through an intuitive web interface.

## Features

- **Model Training**: Train multiple ML models (Logistic Regression, Random Forest, XGBoost) and automatically select the best one
- **Single Prediction**: Predict loan eligibility for individual applicants through a web form
- **Batch Prediction**: Upload CSV files with multiple applicants and get predictions for all
- **Modern UI**: Clean, responsive web interface with real-time feedback
- **Model Persistence**: Trained models are saved and automatically loaded

## Project Structure

```
loan_eligibility_project/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── data/                  # Data directory
│   ├── loan_data.csv      # Training data
│   ├── new_applicants.csv # Sample batch prediction file
│   └── predictions.csv   # Output predictions
├── models/                # Saved models
│   └── best_model.pkl    # Trained model
├── templates/             # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── train.html
│   ├── predict_batch.html
│   ├── result.html
│   └── batch_results.html
└── static/                # Static files
    └── style.css         # CSS styles
```

## Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Flask application**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to**
   ```
   http://localhost:5000
   ```

3. **Train the model** (first time)
   - Click on "Train Model" in the navigation
   - Click "Start Training"
   - Wait for training to complete (this may take a few minutes)
   - The best model will be automatically saved

4. **Make predictions**
   - **Single Prediction**: Fill out the form on the home page and click "Predict Eligibility"
   - **Batch Prediction**: Go to "Batch Prediction", upload a CSV file with applicant data, and download results

## CSV Format for Batch Prediction

The CSV file should contain the following columns:
- `Gender` (Male/Female)
- `Married` (Yes/No)
- `Dependents` (0/1/2/3+)
- `Education` (Graduate/Not Graduate)
- `Self_Employed` (Yes/No)
- `ApplicantIncome` (numeric)
- `CoapplicantIncome` (numeric)
- `LoanAmount` (numeric, optional)
- `Loan_Amount_Term` (numeric, optional)
- `Credit_History` (1 for good, 0 for bad, optional)
- `Property_Area` (Urban/Rural/Semiurban)

Optional columns can be left empty.

## API Endpoints

- `GET /` - Home page with prediction form
- `GET /train` - Training page
- `POST /train` - Train the model
- `POST /predict` - Predict loan eligibility for single applicant
- `GET /predict_batch` - Batch prediction page
- `POST /predict_batch` - Process batch predictions
- `GET /download_predictions` - Download predictions CSV

## Technologies Used

- **Flask**: Web framework
- **scikit-learn**: Machine learning library
- **XGBoost**: Gradient boosting framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **joblib**: Model serialization

## Model Information

The application trains three models:
1. **Logistic Regression**: Linear classification model
2. **Random Forest**: Ensemble of decision trees
3. **XGBoost**: Gradient boosting classifier

The best model is selected based on F1 score and automatically saved for predictions.

## Notes

- The application runs in debug mode by default. For production, set `debug=False` in `app.py`
- Change the `secret_key` in `app.py` for production use
- The model is loaded automatically on startup if it exists
- Training data should be in `data/loan_data.csv` with a `Loan_Status` column (Y/N)

## License

This project is open source and available for educational purposes.

