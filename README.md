
# Bank Transaction Fraud Detection

An end-to-end ML project for detecting fraudulent transactions using machine learning.
Built with Python, FastAPI, and Docker.

## Problem Description
Financial institutions lose billions yearly to fraudulent transactions. __[bank transaction dataset](https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection?resource=download)__
The goal of this project is to develop a machine learning model that identifies potentially fraudulent transactions using historical behavioral data.

The system is designed to:

- Load and preprocess raw transaction data  
- Perform cleaning, handling missing values, and scaling numerical features  
- Engineer additional predictive features based on user behavior and transaction patterns  
- Train multiple machine learning models and compare their performance  
- Automatically select and export the best-performing model  
- Provide a fully functional **FastAPI web service** for real-time predictions  
- Offer an optional Docker-based deployment workflow for portability

This project demonstrates strong understanding of data processing, feature engineering, model training, backend model serving, and deployment fundamentals. The repository is structured to follow best-practice ML engineering principles, making it easy to extend, reproduce, and deploy.


## Features
- Data cleaning and preprocessing
- Feature engineering
- Model training and evaluation
- REST API for real-time predictions
- Docker containerization

## Notebook: Data Preparation & EDA
Open the notebook:
jupyter notebook notebook.ipynb
The notebook includes:
Data cleaning
Exploratory data analysis
Feature engineering
Model comparisons
Metrics (Accuracy, Precision, Recall, F1-score)

## Deployment

To make the fraud-detection model accessible as a real-time prediction service, a FastAPI web application was developed. The service exposes a `/predict` endpoint that accepts transaction details and returns a fraud probability along with a classification (legitimate vs. fraudulent). The API documentation is automatically generated and available through Swagger UI.

A Docker container was created to ensure consistent execution regardless of the host environment. This container bundles the trained model, preprocessing pipeline, all dependencies, and the FastAPI application.

### Create a virtual environment

I Recommended for Clean Dependency Management

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```
Mac/Linux

```bash
source venv/bin/activate
```

## II Install Dependencies

All project dependencies are listed in requirements.txt.

```bash
pip install -r requirements.txt
```

### III Run the training pipeline

This script loads data, performs feature engineering, trains multiple models, selects the best one, and saves it as best_model.pkl.

```bash
python test_model_trainer.py
```

After running, your models/ folder will contain:

```bash
best_model.pkl → best performing ML model
```

### Deployment Workflow

The following steps were used to prepare, test, and deploy the service:

1. **Implemented FastAPI application** for serving predictions  
2. **Created a Dockerfile** to containerize the API  
3. **Built and tested the Docker image locally** using Docker Desktop  
4. (Optional) **Deployed the container to a cloud service**, such as Azure Container Apps, Render, or Railway  
5. Validated the deployed endpoint using Swagger UI and test JSON payloads

### Accessing the Service

Once deployment is complete, the service can be accessed through:

- **API Base URL:**  
  *([API based url](http://127.0.0.1:8000/docs#/default))*
  The API will return:

  ```bash
  {
  "fraud_prediction": 0 or 1,
  "message": "Legitimate Transaction" or "Fraudulent Transaction Detected"
  }
  ```

Start the prediction API (FastAPI)

From the project root:

```bash
uvicorn app.main:app --reload
```

Once the server starts, you can visit:

```bash
Swagger UI: http://127.0.0.1:8000/docs
```
Prediction endpoint: /predict

Example JSON payload:

```bash
{
  "TransactionAmount": 250.0,
  "CustomerAge": 45,
  "LoginAttempts": 2,
  "TransactionDuration": 15.3,
  "AccountBalance": 1200.5
}
```


## How to Run
Follow the steps below to set up and run the full pipeline on your local machine.
1. Create virtual environment  
2. Install dependencies: `pip install -r requirements.txt` All project dependencies are listed in requirements.txt.
3. Run training: `python src/model_trainer.py` This script loads data, performs feature engineering, trains multiple models, selects the best one, and saves it as best_model.pkl.
4. Launch API: `uvicorn app.main:app --reload`
5. Test with Postman or `curl`

##  References
DataTalksClub (2025), Machine Learning Zoomcamp Course Materials. Available at: https://github.com/DataTalksClub/machine-learning-zoomcamp.
