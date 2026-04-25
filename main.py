from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from utils.config import APP_NAME, VERSION, SECRET_KEY_TOKEN, preprocessor, target_encoder, lgb_model
from utils.ApartmentData import ApartmentData
from utils.inference import predict_new

app = FastAPI(title=APP_NAME, version=VERSION)

@app.get('/', tags=['General'])
async def home():
    return {
        'msg': f'Welcome to the {APP_NAME} API v{VERSION}'
    }

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != SECRET_KEY_TOKEN:
        raise HTTPException(status_code=403, detail="You are not authorized to use this API")
    return api_key

@app.post('/models/lgb_baseline', tags=['LightGBM'])
async def lgb_predict(data: ApartmentData, api_key: str = Depends(verify_api_key)) -> dict:
    if lgb_model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded properly.")
    
    try:
        result = predict_new(data=data, preprocessor=preprocessor, target_encoder=target_encoder, model=lgb_model)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
