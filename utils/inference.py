import pandas as pd
import numpy as np
import re

def predict_new(data, preprocessor, target_encoder, model):
    data_dict = data.model_dump(by_alias=True)
    
    # Extract categorical fields
    furnished = data_dict.pop('furnished', 'unknown').lower()
    ownership = data_dict.pop('ownership', 'unknown').lower()
    payment_option = data_dict.pop('payment_option', 'unknown').lower()
    completion_status = data_dict.pop('completion_status', 'unknown').lower()
    seller_type = data_dict.pop('seller_type', 'unknown').lower()
    
    # Initialize OHE fields with spaces as expected by TE
    ohe_cols = [
        'furnished_no', 'furnished_unknown', 'furnished_yes',
        'ownership_primary', 'ownership_resale',
        'payment_option_cash', 'payment_option_cash or installment', 'payment_option_installment',
        'completion_status_off-plan', 'completion_status_ready',
        'seller_type_agency', 'seller_type_individual'
    ]
    for col in ohe_cols:
        data_dict[col] = 0.0
        
    if f"furnished_{furnished}" in data_dict: data_dict[f"furnished_{furnished}"] = 1.0
    if f"ownership_{ownership}" in data_dict: data_dict[f"ownership_{ownership}"] = 1.0
    if f"payment_option_{payment_option}" in data_dict: data_dict[f"payment_option_{payment_option}"] = 1.0
    if f"completion_status_{completion_status}" in data_dict: data_dict[f"completion_status_{completion_status}"] = 1.0
    if f"seller_type_{seller_type}" in data_dict: data_dict[f"seller_type_{seller_type}"] = 1.0

    # Feature Engineering
    area = float(data_dict.get('area_numeric', 0.0))
    beds = float(data_dict.get('bedrooms', 0.0))
    baths = float(data_dict.get('bathrooms', 0.0))
    
    safe_beds = beds if beds != 0 else 0.5
    safe_baths = baths if baths != 0 else 0.5
    
    data_dict['area_per_bedroom'] = area / safe_beds
    data_dict['area_per_bathroom'] = area / safe_baths
    data_dict['bed_bath_ratio'] = beds / safe_baths
    data_dict['total_rooms'] = beds + baths
    
    amenity_cols = [
        'Electricity Meter', 'Water Meter', 'Natural Gas', 'Security', 'Covered Parking', 
        'Pets Allowed', 'Landline', 'Balcony', 'Private Garden', 'Pool', 
        'Built in Kitchen Appliances', 'Elevator', 'Central A/C & heating', 'Maids Room', 'roof'
    ]
    data_dict['amenity_score'] = sum(float(data_dict.get(c, 0.0)) for c in amenity_cols)
    
    comp = data_dict.get('compound_name', 'unknown').lower()
    data_dict['is_compound'] = 1.0 if comp != 'unknown' else 0.0
    
    del_date = float(data_dict.get('delivery_date', 0.0))
    if del_date <= 0.5:
        data_dict['delivery_bucket'] = 0.0
    elif del_date <= 12.0:
        data_dict['delivery_bucket'] = 1.0
    elif del_date <= 24.0:
        data_dict['delivery_bucket'] = 2.0
    else:
        data_dict['delivery_bucket'] = 3.0

    data_dict['area_sqm'] = area
    data_dict.pop('price_per_sqm', None)

    for k, v in data_dict.items():
        if isinstance(v, bool):
            data_dict[k] = float(v)

    # The 48 expected columns by Target Encoder
    te_cols = [
        'bedrooms', 'bathrooms', 'area_sqm', 'area_numeric', 'property_type', 'city', 'governorate', 
        'latitude', 'longitude', 'finish_type', 'seller_name', 'view_type', 'compound_name', 
        'delivery_date', 'Electricity Meter', 'Water Meter', 'Natural Gas', 'Security', 
        'Covered Parking', 'Pets Allowed', 'Landline', 'Balcony', 'Private Garden', 'Pool', 
        'Built in Kitchen Appliances', 'Elevator', 'Central A/C & heating', 'Maids Room', 'roof', 
        'furnished_no', 'furnished_unknown', 'furnished_yes', 'ownership_primary', 'ownership_resale', 
        'payment_option_cash', 'payment_option_cash or installment', 'payment_option_installment', 
        'completion_status_off-plan', 'completion_status_ready', 'seller_type_agency', 'seller_type_individual',
        'area_per_bedroom', 'area_per_bathroom', 'bed_bath_ratio', 'total_rooms', 'amenity_score', 'is_compound', 'delivery_bucket'
    ]
    
    te_data = {col: data_dict[col] for col in te_cols}
    X_new = pd.DataFrame([te_data])
    
    if target_encoder:
        X_encoded = target_encoder.transform(X_new)
    else:
        X_encoded = X_new
        
    if preprocessor and hasattr(preprocessor, 'feature_names_in_'):
        scale_cols = preprocessor.feature_names_in_
        X_encoded[scale_cols] = preprocessor.transform(X_encoded[scale_cols])
        
    # LightGBM requires column names without spaces or special characters
    # Usually it converts spaces to underscores
    lgb_cols = model.feature_name()
    
    # Rename X_encoded columns to match LightGBM perfectly
    rename_map = {}
    for col in X_encoded.columns:
        clean_col = col.replace(' ', '_')
        if clean_col in lgb_cols:
            rename_map[col] = clean_col
        else:
            rename_map[col] = col
    
    X_encoded = X_encoded.rename(columns=rename_map)
    
    X_final = X_encoded[lgb_cols]
    
    y_pred = model.predict(X_final)
    pred_val = float(y_pred[0])
    if pred_val < 100:
        pred_val = float(np.expm1(pred_val))
    
    return {
        "Prediction": pred_val,
        "Currency": "EGP"
    }
