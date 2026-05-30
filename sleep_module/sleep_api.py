from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from functools import lru_cache
import pandas as pd
import joblib

# Access through-> uvicorn sleep_module.sleep_api:app --reload --port 8001

# 1. Initialize FastAPI App
app = FastAPI(
    title="AstroMind Sleep Architecture Diagnostic API",
    description="Analyzes smartwatch vitals via XGBoost to detect clinical sleep disorders.",
    version="1.0.0"
)

# 2. Define the Relative Path to the Model Directory
MODEL_DIR = Path(__file__).parent / "saved_sleep_model"

# 3. Pydantic Schema for Smartwatch Incoming Data
class AstronautVitals(BaseModel):
    gender: str = Field(..., example="Male", description="Male or Female")
    age: int = Field(..., example=34)
    occupation: str = Field(..., example="Astronaut", description="Used as a proxy for environmental stress")
    sleep_duration: float = Field(..., example=5.8, description="Total hours slept")
    quality_of_sleep: int = Field(..., example=4, description="Subjective sleep rating from 1 to 10")
    physical_activity_level: int = Field(..., example=75, description="Minutes of daily activity")
    stress_level: int = Field(..., example=8, description="Perceived cognitive stress level from 1 to 10")
    bmi_category: str = Field(..., example="Normal", description="Normal, Overweight, or Obese")
    heart_rate: int = Field(..., example=82, description="Nocturnal resting heart rate in BPM")
    daily_steps: int = Field(..., example=11000)
    systolic_bp: int = Field(..., example=135, description="Systolic Blood Pressure")
    diastolic_bp: int = Field(..., example=88, description="Diastolic Blood Pressure")


# 4. Lazy Loader Cache (Reads from disk only ONCE on first request)
@lru_cache(maxsize=1)
def get_sleep_pipeline():
    """Safely loads model weights and dependencies into memory."""
    model_path = MODEL_DIR / "xgboost_sleep_model.pkl"
    encoder_path = MODEL_DIR / "target_encoder.pkl"
    columns_path = MODEL_DIR / "feature_columns.pkl"
    
    # Safety Check: Stop the server if the folder is missing or misnamed
    if not (model_path.exists() and encoder_path.exists() and columns_path.exists()):
        raise RuntimeError(
            f"❌ Critical Error: Sleep model files not found in {MODEL_DIR}. "
            "Please ensure you ran the export script in your notebook."
        )
        
    xgb_model = joblib.load(model_path)
    target_encoder = joblib.load(encoder_path)
    feature_order = joblib.load(columns_path)
    
    print("🌲 [SUCCESS] XGBoost Sleep Model loaded into RAM!")
    return xgb_model, target_encoder, feature_order


# 5. Production Feature Engineering Pipeline
def engineer_features_production(input_df):
    """Replicates the identical feature extraction transformations used during training."""
    df_feat = input_df.copy()
    
    # Map environmental/operational stress proxy
    stress_map = {'Scientist': 5, 'Astronaut': 9, 'Doctor': 8}
    df_feat['Occupational_Stress_Index'] = df_feat['Occupation'].map(stress_map).fillna(5)
    
    # Ordinal and Binary Maps
    df_feat['BMI_Score'] = df_feat['BMI Category'].map({'Normal': 0, 'Overweight': 1, 'Obese': 2}).fillna(0)
    df_feat['Is_Male'] = df_feat['Gender'].map({'Male': 1, 'Female': 0}).fillna(1)
    
    # Multi-variable Proxy Biomarkers
    df_feat['Restorative_Rest_Score'] = df_feat['Sleep Duration'] * df_feat['Quality of Sleep']
    df_feat['Nocturnal_Strain_Index'] = df_feat['Heart Rate'] / (df_feat['Sleep Duration'] + 1e-5)
    df_feat['Cardio_Load_Factor'] = df_feat['Systolic_BP'] / (df_feat['Heart Rate'] + 1e-5)
    df_feat['Step_Intensity'] = df_feat['Daily Steps'] / (df_feat['Physical Activity Level'] + 1e-5)
    
    # Drop original structural text columns
    return df_feat.drop(columns=['Gender', 'Occupation', 'BMI Category'])


# 6. The API Diagnostic Endpoint
@app.post("/sleep/diagnose")
async def diagnose_sleep(vitals: AstronautVitals):
    try:
        # Step A: Fetch pipeline artifacts from cache
        xgb_model, target_encoder, feature_order = get_sleep_pipeline()
        
        # Step B: Map incoming JSON request keys to match training dataframe headers
        raw_data = {
            'Gender': vitals.gender,
            'Age': vitals.age,
            'Occupation': vitals.occupation,
            'Sleep Duration': vitals.sleep_duration,
            'Quality of Sleep': vitals.quality_of_sleep,
            'Physical Activity Level': vitals.physical_activity_level,
            'Stress Level': vitals.stress_level,
            'BMI Category': vitals.bmi_category,
            'Heart Rate': vitals.heart_rate,
            'Daily Steps': vitals.daily_steps,
            'Systolic_BP': vitals.systolic_bp,
            'Diastolic_BP': vitals.diastolic_bp
        }
        
        df_raw = pd.DataFrame([raw_data])
        
        # Step C: Extract proxies and enforce feature ordering sequence
        df_processed = engineer_features_production(df_raw)
        df_processed = df_processed[feature_order]
        
        # Step D: Model inference execution
        pred_idx = xgb_model.predict(df_processed)[0]
        pred_label = target_encoder.inverse_transform([pred_idx])[0]
        probabilities = xgb_model.predict_proba(df_processed)[0]
        
        # Step E: Return structured clean JSON response
        return {
            "status": "success",
            "diagnosis": pred_label,
            "metrics": {
                "restorative_rest_score": float(df_processed['Restorative_Rest_Score'].iloc[0]),
                "nocturnal_strain_index": float(df_processed['Nocturnal_Strain_Index'].iloc[0])
            },
            "confidence_matrix": {
                "None": round(float(probabilities[0]), 4),
                "Sleep_Apnea": round(float(probabilities[1]), 4),
                "Insomnia": round(float(probabilities[2]), 4)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine failure: {str(e)}")

# 7. Root Status Check Endpoint
@app.get("/")
def read_root():
    return {"module": "AstroMind Sleep Module", "status": "operational"}