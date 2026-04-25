import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Dubizzle Egypt Real Estate Price Predictor (SOTA Tabular Models)\n",
    "\n",
    "This notebook trains State-Of-The-Art (SOTA) machine learning and deep learning models to predict apartment prices based on scraped Dubizzle data. For tabular data like this, Gradient Boosted Trees (CatBoost, XGBoost, LightGBM) typically outperform pure Deep Learning, but we will train both and stack them for the absolute best results.\n",
    "\n",
    "**Please upload your `dubizzle_cleaned.csv` or `dubizzle_cleaned.json` to the Colab environment before running this.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -q catboost lightgbm xgboost shap tensorflow scikit-learn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.impute import SimpleImputer\n",
    "\n",
    "# ML Models\n",
    "import xgboost as xgb\n",
    "import lightgbm as lgb\n",
    "from catboost import CatBoostRegressor\n",
    "\n",
    "# Deep Learning\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense, Dropout, BatchNormalization\n",
    "from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Loading and Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data - Ensure you've uploaded 'dubizzle_cleaned.json' to Colab\n",
    "try:\n",
    "    df = pd.read_json('dubizzle_cleaned.json')\n",
    "except FileNotFoundError:\n",
    "    print(\"Please upload dubizzle_cleaned.json to the workspace.\")\n",
    "\n",
    "print(f\"Loaded {len(df)} records.\")\n",
    "display(df.head(2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess_data(df):\n",
    "    data = df.copy()\n",
    "    \n",
    "    # Select relevant features for prediction\n",
    "    num_cols = ['bedrooms', 'bathrooms', 'area_numeric', 'latitude', 'longitude']\n",
    "    # Clean numeric columns (convert to float, handle weird strings if any)\n",
    "    for col in num_cols:\n",
    "        data[col] = pd.to_numeric(data[col], errors='coerce')\n",
    "        \n",
    "    # Categorical columns\n",
    "    cat_cols = ['city', 'region', 'property_type', 'completion_status', 'ownership', \n",
    "                'payment_option', 'finish_type', 'view_type']\n",
    "    \n",
    "    for col in cat_cols:\n",
    "        data[col] = data[col].fillna('Unknown').astype(str)\n",
    "        \n",
    "    # Boolean features (from amenities)\n",
    "    # If you used your extract_features script, you might have has_garden, has_pool etc.\n",
    "    # Let's extract them from the amenities string if they exist, or use direct bool columns if available.\n",
    "    bool_cols = [c for c in data.columns if c.startswith('has_')]\n",
    "    if not bool_cols:\n",
    "        # Fallback if bools weren't saved cleanly, parse from 'amenities'\n",
    "        data['amenities'] = data['amenities'].fillna('').astype(str).str.lower()\n",
    "        data['has_pool'] = data['amenities'].str.contains('pool').astype(int)\n",
    "        data['has_garden'] = data['amenities'].str.contains('garden').astype(int)\n",
    "        data['has_security'] = data['amenities'].str.contains('security').astype(int)\n",
    "        data['has_parking'] = data['amenities'].str.contains('parking').astype(int)\n",
    "        bool_cols = ['has_pool', 'has_garden', 'has_security', 'has_parking']\n",
    "    else:\n",
    "        for col in bool_cols:\n",
    "            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)\n",
    "            \n",
    "    # We need the target variable\n",
    "    data = data.dropna(subset=['price_numeric', 'area_numeric'])\n",
    "    \n",
    "    # Target variable: Log scale because real estate prices are heavily right-skewed\n",
    "    data['target_log_price'] = np.log1p(data['price_numeric'])\n",
    "    \n",
    "    features = num_cols + cat_cols + bool_cols\n",
    "    \n",
    "    return data, features, num_cols, cat_cols, bool_cols\n",
    "\n",
    "data, features, num_cols, cat_cols, bool_cols = preprocess_data(df)\n",
    "\n",
    "X = data[features]\n",
    "y = data['target_log_price']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "print(f\"Training set shape: {X_train.shape}\")\n",
    "print(f\"Test set shape: {X_test.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Setting up the Pipeline\n",
    "We use Sklearn Pipelines to scale numerics and encode categoricals for our basic and deep learning models. \n",
    "*(Note: CatBoost handles categoricals natively, but LightGBM/XGB/DL require encoding).* "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "num_transformer = Pipeline(steps=[\n",
    "    ('imputer', SimpleImputer(strategy='median')),\n",
    "    ('scaler', StandardScaler())\n",
    "])\n",
    "\n",
    "cat_transformer = Pipeline(steps=[\n",
    "    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),\n",
    "    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))\n",
    "])\n",
    "\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', num_transformer, num_cols),\n",
    "        ('cat', cat_transformer, cat_cols),\n",
    "        ('bool', 'passthrough', bool_cols)\n",
    "    ])\n",
    "\n",
    "# Preprocess data for generic models\n",
    "X_train_processed = preprocessor.fit_transform(X_train)\n",
    "X_test_processed = preprocessor.transform(X_test)\n",
    "\n",
    "print(f\"Processed features shape: {X_train_processed.shape}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Training SOTA Tree-Based Models\n",
    "For tabular data, trees rule. We train LightGBM, XGBoost, and CatBoost."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def evaluate_model(name, y_true_log, y_pred_log):\n",
    "    # Convert back from log scale\n",
    "    y_true = np.expm1(y_true_log)\n",
    "    y_pred = np.expm1(y_pred_log)\n",
    "    \n",
    "    mae = mean_absolute_error(y_true, y_pred)\n",
    "    rmse = np.sqrt(mean_squared_error(y_true, y_pred))\n",
    "    r2 = r2_score(y_true, y_pred) # Better to calculate R2 on log scale, but let's check absolute\n",
    "    r2_log = r2_score(y_true_log, y_pred_log)\n",
    "    \n",
    "    print(f\"--- {name} ---\")\n",
    "    print(f\"R2 (log space): {r2_log:.4f}\")\n",
    "    print(f\"MAE (EGP): {mae:,.0f}\")\n",
    "    print(f\"RMSE (EGP): {rmse:,.0f}\\n\")\n",
    "    return y_pred_log"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# LightGBM\n",
    "lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8, random_state=42)\n",
    "lgb_model.fit(X_train_processed, y_train, \n",
    "              eval_set=[(X_test_processed, y_test)],\n",
    "              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])\n",
    "\n",
    "lgb_preds = lgb_model.predict(X_test_processed)\n",
    "evaluate_model(\"LightGBM\", y_test, lgb_preds);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# XGBoost\n",
    "xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=8, random_state=42, \n",
    "                             objective='reg:squarederror', early_stopping_rounds=50)\n",
    "xgb_model.fit(X_train_processed, y_train, eval_set=[(X_test_processed, y_test)], verbose=False)\n",
    "\n",
    "xgb_preds = xgb_model.predict(X_test_processed)\n",
    "evaluate_model(\"XGBoost\", y_test, xgb_preds);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# CatBoost (Native Categorical Handling - Often Best)\n",
    "# We feed it the original unprocessed dataframe and tell it which columns are categorical\n",
    "cat_features_indices = [X_train.columns.get_loc(c) for c in cat_cols]\n",
    "\n",
    "# Impute numerical missing values for CatBoost (it handles NaNs in cats, but numbers need filling)\n",
    "X_train_cb = X_train.copy()\n",
    "X_test_cb = X_test.copy()\n",
    "for c in num_cols:\n",
    "    med = X_train_cb[c].median()\n",
    "    X_train_cb[c] = X_train_cb[c].fillna(med)\n",
    "    X_test_cb[c] = X_test_cb[c].fillna(med)\n",
    "\n",
    "cb_model = CatBoostRegressor(iterations=1500, learning_rate=0.05, depth=8, loss_function='RMSE', random_seed=42, verbose=0)\n",
    "cb_model.fit(X_train_cb, y_train, cat_features=cat_features_indices, eval_set=(X_test_cb, y_test), early_stopping_rounds=50)\n",
    "\n",
    "cb_preds = cb_model.predict(X_test_cb)\n",
    "evaluate_model(\"CatBoost\", y_test, cb_preds);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Deep Learning (Neural Network)\n",
    "While trees are usually better for structured data, a well-tuned NN can capture different interactions and provides diversity for an ensemble."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_dl_model(input_dim):\n",
    "    model = Sequential([\n",
    "        Dense(256, activation='relu', input_dim=input_dim),\n",
    "        BatchNormalization(),\n",
    "        Dropout(0.3),\n",
    "        \n",
    "        Dense(128, activation='relu'),\n",
    "        BatchNormalization(),\n",
    "        Dropout(0.2),\n",
    "        \n",
    "        Dense(64, activation='relu'),\n",
    "        Dense(1, activation='linear')\n",
    "    ])\n",
    "    \n",
    "    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])\n",
    "    return model\n",
    "\n",
    "dl_model = build_dl_model(X_train_processed.shape[1])\n",
    "\n",
    "callbacks = [\n",
    "    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),\n",
    "    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)\n",
    "]\n",
    "\n",
    "history = dl_model.fit(X_train_processed, y_train, \n",
    "                       validation_data=(X_test_processed, y_test),\n",
    "                       epochs=100, batch_size=64, callbacks=callbacks, verbose=0)\n",
    "\n",
    "# Plot training history\n",
    "plt.plot(history.history['loss'], label='Train Loss')\n",
    "plt.plot(history.history['val_loss'], label='Val Loss')\n",
    "plt.title('Deep Learning Training History')\n",
    "plt.legend()\n",
    "plt.show()\n",
    "\n",
    "dl_preds = dl_model.predict(X_test_processed).flatten()\n",
    "evaluate_model(\"Deep Learning (Keras)\", y_test, dl_preds);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Stacking / Ensemble\n",
    "We average the predictions of our best models to achieve the absolute SOTA. The wisdom of the crowd smooths out individual model errors."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Simple average ensemble\n",
    "# Based on typical performance, CatBoost and LightGBM get slightly higher weights.\n",
    "ensemble_preds = (cb_preds * 0.4) + (lgb_preds * 0.3) + (xgb_preds * 0.2) + (dl_preds * 0.1)\n",
    "\n",
    "print(\"========== FINAL ENSEMBLE PERFORMANCE ==========\")\n",
    "evaluate_model(\"Weighted Ensemble\", y_test, ensemble_preds)\n",
    "\n",
    "# Visualizing Predictions vs Reality (Ensemble)\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.scatter(np.expm1(y_test), np.expm1(ensemble_preds), alpha=0.3, color='purple')\n",
    "plt.plot([0, 50000000], [0, 50000000], 'r--')\n",
    "plt.xlim(0, 30000000)\n",
    "plt.ylim(0, 30000000)\n",
    "plt.xlabel('Actual Price (EGP)')\n",
    "plt.ylabel('Predicted Price (EGP)')\n",
    "plt.title('Ensemble Model: Predicted vs Actual Prices')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Model Interpretation (SHAP)\n",
    "Let's see what features drive property prices in Egypt using CatBoost's feature importance."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import shap\n",
    "\n",
    "explainer = shap.Explainer(cb_model)\n",
    "# Sample a subset for faster shap calculation\n",
    "shap_values = explainer(X_test_cb.sample(1000, random_state=42))\n",
    "\n",
    "shap.plots.beeswarm(shap_values, max_display=15)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Save Models and Preprocessor\n",
    "Saving the trained components to disk so they can be loaded later for production inference."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib\n",
    "import os\n",
    "\n",
    "os.makedirs('saved_models', exist_ok=True)\n",
    "\n",
    "# Save the preprocessor pipeline\n",
    "joblib.dump(preprocessor, 'saved_models/dubizzle_preprocessor.pkl')\n",
    "\n",
    "# Save the individual SOTA models\n",
    "lgb_model.booster_.save_model('saved_models/lgb_model.txt')\n",
    "xgb_model.save_model('saved_models/xgb_model.json')\n",
    "cb_model.save_model('saved_models/cb_model.cbm')\n",
    "\n",
    "# Save Deep Learning model\n",
    "dl_model.save('saved_models/dl_model.keras')\n",
    "\n",
    "print(\"✅ Success! Preprocessor and models saved to the 'saved_models' directory.\")\n",
    "print(\"Make sure to download this folder to your local machine for deployment.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("price_prediction_sota.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully: price_prediction_sota.ipynb")
