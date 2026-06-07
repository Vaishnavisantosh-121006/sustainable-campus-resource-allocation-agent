import os
import shutil
import pytest
import pandas as pd
from src.sustainable_campus.ai.dataset_generator import generate_campus_dataset
from src.sustainable_campus.ai.demand_prediction import train_and_save_models, evaluate_regression_model

TEMP_DIR = "temp_test_dir"
TEMP_DATA = os.path.join(TEMP_DIR, "test_demand.csv")
TEMP_MODELS = os.path.join(TEMP_DIR, "models")

@pytest.fixture(scope="module", autouse=True)
def setup_temp_dir():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    yield
    # Clean up after tests run
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

def test_dataset_generation():
    df = generate_campus_dataset(TEMP_DATA)
    assert os.path.exists(TEMP_DATA)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 8760
    assert "classroom_demand" in df.columns
    assert "laboratory_demand" in df.columns
    assert "hostel_demand" in df.columns

def test_model_training():
    results = train_and_save_models(TEMP_DATA, TEMP_MODELS)
    assert "classroom" in results
    assert "laboratory" in results
    assert "hostel" in results
    
    assert os.path.exists(os.path.join(TEMP_MODELS, "classroom_demand_model.joblib"))
    assert os.path.exists(os.path.join(TEMP_MODELS, "laboratory_demand_model.joblib"))
    assert os.path.exists(os.path.join(TEMP_MODELS, "hostel_demand_model.joblib"))

def test_evaluation_helper():
    import numpy as np
    y_true = np.array([100.0, 150.0, 200.0])
    y_pred = np.array([105.0, 145.0, 205.0])
    
    metrics = evaluate_regression_model(y_true, y_pred)
    assert metrics["MAE"] == pytest.approx(5.0)
    assert metrics["RMSE"] == pytest.approx(5.0)
    assert metrics["R2"] > 0.9
