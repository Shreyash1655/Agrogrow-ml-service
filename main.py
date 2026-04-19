import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib

# Initialize FastAPI
app = FastAPI(title="AgroGROW AI Service")

# IMPORTANT: Allow React Native / Spring Boot to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the models folder exists to save our trained brains
os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------
# 1. DATA LAYER
# ---------------------------------------------------------
def fetch_data_from_db(commodity: str, mandi: str):
    """
    In a full production environment, this queries your PostgreSQL DB.
    For now, it generates 45 days of highly realistic baseline data 
    so the AI can train on ANY crop you select in the UI.
    """
    # Base prices per Quintal
    base_prices = {
        "Cashew": 14500,
        "Coconut": 3200,
        "Arecanut": 45000,
        "Black Pepper": 52000,
        "Mango": 6500
    }
    
    start_price = base_prices.get(commodity, 5000)
    dates = [datetime.today() - timedelta(days=x) for x in range(45, 0, -1)]
    
    # Generate 45 days of fluctuating prices (trend + noise)
    prices = []
    current_price = start_price
    for _ in range(45):
        # +/- 2% daily volatility
        change = current_price * np.random.uniform(-0.02, 0.02) 
        current_price += change
        prices.append(round(current_price))
        
    df = pd.DataFrame({
        "date": dates,
        "modal_price": prices
    })
    return df

# ---------------------------------------------------------
# 2. AI TRAINING LAYER (LSTM)
# ---------------------------------------------------------
def train_and_save_model(commodity: str, df: pd.DataFrame):
    """
    Trains the LSTM model and saves it. This takes time, 
    but only happens ONCE per crop.
    """
    print(f"🧠 [TRAINING STARTED] Building new AI for {commodity}...")
    
    # Prepare Data
    dataset = df['modal_price'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Create sequences (Look back 7 days to predict the next)
    look_back = 7
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])
        
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the Model
    model.fit(X, y, epochs=50, batch_size=8, verbose=0)
    
    # Save the "Brain" and the "Scaler"
    model.save(f"models/model_{commodity}.h5")
    joblib.dump(scaler, f"models/scaler_{commodity}.pkl")
    
    print(f"✅ [TRAINING COMPLETE] Brain saved for {commodity}.")
    return model, scaler

# ---------------------------------------------------------
# 3. API ENDPOINT (The Bridge to React Native/Spring Boot)
# ---------------------------------------------------------
@app.get("/api/arbitrage/test-ml")
def get_market_intelligence(commodity: str = "Cashew", mandi: str = "Panaji"):
    try:
        model_path = f"models/model_{commodity}.h5"
        scaler_path = f"models/scaler_{commodity}.pkl"
        
        # Fetch the 45-day history
        df = fetch_data_from_db(commodity, mandi)
        current_price = df['modal_price'].iloc[-1]
        
        # THE SPEED DEMON FIX: Load instead of Train!
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            print(f"⚡ [FAST LOAD] Waking up saved AI for {commodity}...")
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
        else:
            # Slow path (First run only)
            model, scaler = train_and_save_model(commodity, df)
            
        # ---------------------------------------------------------
        # Predict the next 7 days
        # ---------------------------------------------------------
        look_back = 7
        # Get the last 7 days of known data
        last_7_days = df['modal_price'].values[-look_back:].reshape(-1, 1)
        last_7_scaled = scaler.transform(last_7_days)
        
        current_batch = last_7_scaled.reshape(1, look_back, 1)
        predicted_prices = []
        
        # Loop to predict 7 days into the future
        for _ in range(7):
            pred_scaled = model.predict(current_batch, verbose=0)[0]
            predicted_prices.append(pred_scaled[0])
            
            # Slide the window forward: remove oldest day, append new prediction
            current_batch = np.append(current_batch[:, 1:, :], [[pred_scaled]], axis=1)
            
        # Convert scaled numbers back to real Rupees (₹)
        predicted_prices_real = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
        
        # Format the output for the React Native UI
        predictions_list = []
        base_date = datetime.today()
        for i in range(7):
            pred_date = (base_date + timedelta(days=i+1)).strftime('%Y-%m-%d')
            predictions_list.append({
                "date": pred_date,
                "modal_price": float(predicted_prices_real[i][0])
            })
            
        return {
            "commodity": commodity,
            "mandi_name": mandi,
            "current_price": float(current_price),
            "confidence_score": round(np.random.uniform(0.78, 0.92), 2), # Simulated AI confidence
            "predictions": predictions_list
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the server automatically if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)