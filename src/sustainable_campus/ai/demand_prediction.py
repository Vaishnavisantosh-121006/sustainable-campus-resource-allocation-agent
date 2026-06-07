import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_regression_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes regression evaluation metrics: MAE, RMSE, and R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

def train_and_save_models(data_path: str, models_dir: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Trains and saves energy demand prediction models for classrooms, laboratories, and hostels.
    Compares Linear Regression and Random Forest Regressor.
    """
    logger.info(f"Training prediction models using data from {data_path}")
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    df = pd.read_csv(data_path)
    
    agent_types = ["classroom", "laboratory", "hostel"]
    results = {}

    for agent in agent_types:
        logger.info(f"Training models for {agent}...")
        
        # Select agent-specific columns
        occ_col = f"{agent}_occupancy"
        prev_col = f"{agent}_prev_demand"
        target_col = f"{agent}_demand"
        
        features = ["day_of_week", "hour", "temperature", "is_holiday", "is_exam_period", occ_col, prev_col]
        
        # Rename agent-specific columns to generic feature names for model reuse/consistency
        X = df[features].copy()
        X.columns = ["day_of_week", "hour", "temperature", "is_holiday", "is_exam_period", "occupancy", "prev_demand"]
        y = df[target_col].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 1. Train Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)
        lr_metrics = evaluate_regression_model(y_test, lr_preds)
        
        # 2. Train Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_metrics = evaluate_regression_model(y_test, rf_preds)

        logger.info(f"{agent.capitalize()} - Linear Regression: MAE={lr_metrics['MAE']:.2f}, R2={lr_metrics['R2']:.3f}")
        logger.info(f"{agent.capitalize()} - Random Forest: MAE={rf_metrics['MAE']:.2f}, R2={rf_metrics['R2']:.3f}")

        # Determine best model (based on R2 score)
        best_model = rf if rf_metrics["R2"] > lr_metrics["R2"] else lr
        best_model_name = "random_forest" if rf_metrics["R2"] > lr_metrics["R2"] else "linear_regression"
        
        # Save best model
        model_filename = f"{agent}_demand_model.joblib"
        model_save_path = os.path.join(models_dir, model_filename)
        joblib.dump(best_model, model_save_path)
        logger.info(f"Saved best model ({best_model_name}) to {model_save_path}")

        results[agent] = {
            "linear_regression": lr_metrics,
            "random_forest": rf_metrics,
            "best_model": best_model_name
        }

    return results

if __name__ == "__main__":
    train_and_save_models("data/campus_energy_demand.csv", "models/")
