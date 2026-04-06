from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import webbrowser
import threading
import time

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "best_model.pkl"

# Load model if it exists
model = None
if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")


# --------------------------------------------------
# Home Page
# --------------------------------------------------
@app.route('/')
def index():
    """Home page with prediction form"""
    return render_template('index.html', model_loaded=model is not None)


# --------------------------------------------------
# Utility: Get current model info
# --------------------------------------------------
def get_model_info():
    """Get information about the trained model using the stored pipeline"""
    if model is None or not MODEL_PATH.exists():
        return None

    try:
        data_path = DATA_DIR / "loan_data.csv"
        if not data_path.exists():
            return None

        df = pd.read_csv(data_path)
        target_col = "Loan_Status"

        if target_col not in df.columns:
            return None

        # Convert Y/N to 1/0
        df[target_col] = df[target_col].map({"Y": 1, "N": 0})

        # Drop Loan_ID and target
        X = df.drop(columns=[target_col, 'Loan_ID'] if 'Loan_ID' in df.columns else [target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        model_name = type(model.named_steps['model']).__name__

        return {
            "name": model_name,
            "accuracy": round(acc, 3),
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
        }
    except Exception as e:
        print(f"Error getting model info: {e}")
        return None


# --------------------------------------------------
# Train Route
# --------------------------------------------------
@app.route('/train', methods=['GET', 'POST'])
def train():
    """Train the model"""
    if request.method == 'GET':
        model_info = get_model_info()
        return render_template('train.html', best_result=model_info)

    try:
        data_path = DATA_DIR / "loan_data.csv"
        if not data_path.exists():
            flash("Training data file not found!", "error")
            return redirect(url_for('index'))

        df = pd.read_csv(data_path)
        target_col = "Loan_Status"

        if target_col not in df.columns:
            flash(f"Target column '{target_col}' not found in dataset.", "error")
            return redirect(url_for('index'))

        # Convert Y/N → 1/0
        df[target_col] = df[target_col].map({"Y": 1, "N": 0})

        # Features and target
        X = df.drop(columns=[target_col, 'Loan_ID'] if 'Loan_ID' in df.columns else [target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Preprocessing
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="mean"), num_cols),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]), cat_cols)
            ]
        )

        # Candidate models
        models = {
            "LogisticRegression": LogisticRegression(max_iter=300),
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss"
            ),
        }

        best_model = None
        best_score = -1
        best_result = None

        for name, model_obj in models.items():
            pipe = Pipeline([
                ("preprocess", preprocessor),
                ("model", model_obj),
            ])

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)

            result = {
                "name": name,
                "accuracy": round(acc, 3),
                "precision": round(prec, 3),
                "recall": round(rec, 3),
                "f1": round(f1, 3),
            }

            if f1 > best_score:
                best_score = f1
                best_model = pipe
                best_result = result

        # Save best model
        joblib.dump(best_model, MODEL_PATH)
        global model
        model = best_model

        flash(f"Model trained successfully! Best model: {best_result['name']} (F1 Score: {best_result['f1']})", "success")
        return render_template('train.html', best_result=best_result)

    except Exception as e:
        flash(f"Error during training: {str(e)}", "error")
        return redirect(url_for('train'))


# --------------------------------------------------
# Single Prediction Route
# --------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    """Predict loan eligibility for a single applicant (form input)"""
    if model is None:
        flash("Model not trained. Please train the model first.", "error")
        return redirect(url_for('index'))

    try:
        # Read form values
        gender = request.form.get('Gender')
        married = request.form.get('Married')
        dependents = request.form.get('Dependents')
        education = request.form.get('Education')
        self_emp = request.form.get('Self_Employed')

        applicant_income = float(request.form.get('ApplicantIncome') or 0)
        coapp_income = float(request.form.get('CoapplicantIncome') or 0)
        total_income = applicant_income + coapp_income

        # Loan amount entered in RUPEES by user
        loan_amount_rupees = float(request.form.get('LoanAmount') or 0)

        loan_term_str = request.form.get('Loan_Amount_Term') or None
        credit_history_str = request.form.get('Credit_History') or None
        property_area = request.form.get('Property_Area')

        # For the model → convert rupees to thousands
        loan_amount_for_model = loan_amount_rupees / 1000.0 if loan_amount_rupees else 0.0
        loan_term_val = float(loan_term_str) if loan_term_str not in (None, "",) else None
        credit_history_val = float(credit_history_str) if credit_history_str not in (None, "",) else None

        # Data used for model prediction
        model_row = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_emp,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapp_income,
            'LoanAmount': loan_amount_for_model,
            'Loan_Amount_Term': loan_term_val,
            'Credit_History': credit_history_val,
            'Property_Area': property_area
        }

        df = pd.DataFrame([model_row])

        # Ensure column order
        expected_cols = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History', 'Property_Area'
        ]
        df = df[expected_cols]

        # Prediction
        prediction = model.predict(df)[0]           # 1 or 0
        prediction_proba = model.predict_proba(df)[0]

        # Build explanation (simple & realistic)
        explanations = []
        approval_criteria = []

        loan_to_income_ratio = (loan_amount_rupees / total_income * 100) if total_income > 0 and loan_amount_rupees > 0 else None

        # Explanation based on prediction
        if prediction == 1:  # Approved
            if credit_history_val == 1:
                explanations.append("Good credit history strongly supports loan approval.")
            if loan_to_income_ratio is not None and loan_to_income_ratio <= 200:
                explanations.append(f"Loan amount is reasonable compared to income (LTI ≈ {loan_to_income_ratio:.1f}%).")
            if total_income >= 4000:
                explanations.append(f"Total income (₹{total_income:.0f}) is sufficient for this loan amount.")
            if not explanations:
                explanations.append("The overall profile fits typical approved loan applications.")

            approval_criteria.append("Your profile meets the model's approval criteria.")

        else:  # Rejected
            if credit_history_val == 0:
                explanations.append("Poor credit history reduces the chance of loan approval.")
            if loan_to_income_ratio is not None and loan_to_income_ratio > 300:
                explanations.append(f"Loan amount is very high compared to income (LTI ≈ {loan_to_income_ratio:.1f}%).")
            if applicant_income < 3000:
                explanations.append("Applicant income is relatively low for this requested loan amount.")
            if not explanations:
                explanations.append("The model found this profile to be high risk based on multiple factors.")

            # Suggestions to improve
            if credit_history_val != 1:
                approval_criteria.append("Improve credit history (on-time EMI payments, no defaults).")
            if loan_to_income_ratio is not None and loan_to_income_ratio > 200:
                suggested = total_income * 2  # rough safe cap: 200% of monthly income
                approval_criteria.append(
                    f"Consider reducing the loan amount. For your income (₹{total_income:.0f}), "
                    f"a safer loan amount is around ₹{suggested:.0f} or lower."
                )
            if applicant_income < 3000:
                approval_criteria.append("Higher stable income would increase the chances of approval.")

        # Data passed to template for display (keep rupees, original values)
        input_data = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_emp,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapp_income,
            'LoanAmount': loan_amount_rupees,
            'Loan_Amount_Term': loan_term_str,
            'Credit_History': credit_history_str,
            'Property_Area': property_area
        }

        result = {
            "prediction": "Approved" if prediction == 1 else "Rejected",
            "probability_approved": round(prediction_proba[1] * 100, 2),
            "probability_rejected": round(prediction_proba[0] * 100, 2),
            "explanations": explanations,
            "approval_criteria": approval_criteria
        }

        return render_template('result.html', result=result, input_data=input_data)

    except Exception as e:
        flash(f"Prediction error: {str(e)}", "error")
        return redirect(url_for('index'))


# --------------------------------------------------
# Batch Prediction Route
# --------------------------------------------------
@app.route('/predict_batch', methods=['GET', 'POST'])
def predict_batch():
    """Predict loan eligibility for multiple applicants from CSV"""
    if request.method == 'GET':
        return render_template('predict_batch.html', model_loaded=model is not None)

    if model is None:
        flash("Model not trained. Please train the model first.", "error")
        return redirect(url_for('predict_batch'))

    try:
        if 'file' not in request.files:
            flash("No file uploaded", "error")
            return redirect(url_for('predict_batch'))

        file = request.files['file']
        if file.filename == '':
            flash("No file selected", "error")
            return redirect(url_for('predict_batch'))

        # Read CSV
        df = pd.read_csv(file)

        # Keep copy for display/output
        original_df = df.copy()

        # Remove Loan_ID from features if present
        if 'Loan_ID' in df.columns:
            df_features = df.drop(columns=['Loan_ID'])
        else:
            df_features = df.copy()

        # Expected feature columns (in same format as training dataset)
        expected_cols = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
            'Loan_Amount_Term', 'Credit_History', 'Property_Area'
        ]
        df_features = df_features[[col for col in expected_cols if col in df_features.columns]]

        # NOTE: For batch, we assume LoanAmount is already in thousands (same as training data)

        # Predict
        predictions = model.predict(df_features)
        prediction_probas = model.predict_proba(df_features)

        # Build result dataframe for saving & display
        results_df = original_df.copy()
        results_df['Prediction'] = ['Approved' if p == 1 else 'Rejected' for p in predictions]
        results_df['Prob_Approved'] = [round(prob[1] * 100, 2) for prob in prediction_probas]
        results_df['Prob_Rejected'] = [round(prob[0] * 100, 2) for prob in prediction_probas]

        # Save results CSV
        output_path = DATA_DIR / "predictions.csv"
        results_df.to_csv(output_path, index=False)

        results = results_df.to_dict('records')

        flash(f"Predictions completed for {len(results_df)} applicants!", "success")
        return render_template('batch_results.html', results=results, output_path=str(output_path))

    except Exception as e:
        flash(f"Batch prediction error: {str(e)}", "error")
        return redirect(url_for('predict_batch'))


# --------------------------------------------------
# Download predictions.csv
# --------------------------------------------------
@app.route('/download_predictions')
def download_predictions():
    """Download predictions CSV file"""
    predictions_path = DATA_DIR / "predictions.csv"
    if predictions_path.exists():
        return send_file(predictions_path, as_attachment=True, download_name='predictions.csv')
    else:
        flash("Predictions file not found", "error")
        return redirect(url_for('index'))


# --------------------------------------------------
# Auto-open Browser
# --------------------------------------------------
def open_browser():
    """Open the browser after a short delay to allow the server to start"""
    time.sleep(1.5)
    webbrowser.open_new('http://127.0.0.1:5000')


if __name__ == '__main__':
    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    print("=" * 50)
    print("Loan Eligibility Predictor")
    print("=" * 50)
    print("Server starting...")
    print("Opening browser at http://127.0.0.1:5000")
    print("Press CTRL+C to stop the server")
    print("=" * 50)

    app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
