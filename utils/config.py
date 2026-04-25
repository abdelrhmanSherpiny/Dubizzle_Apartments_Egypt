from dotenv import load_dotenv
import os
import joblib
import lightgbm as lgb

load_dotenv(override=True)

APP_NAME = os.getenv('APP_NAME', 'Apartment Price Prediction')
VERSION = os.getenv('VERSION', '1.0.0')
SECRET_KEY_TOKEN = os.getenv('SECRET_KEY_TOKEN', 'default-token')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

try:
    # Load preprocessors
    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'standard_scaler.pkl'))
    target_encoder = joblib.load(os.path.join(MODELS_DIR, 'target_encoder.pkl'))

    # Load LightGBM model
    lgb_model = lgb.Booster(model_file=os.path.join(MODELS_DIR, 'lgb_baseline.txt'))
except Exception as e:
    print(f"Error loading models: {e}")
    preprocessor = None
    target_encoder = None
    lgb_model = None
