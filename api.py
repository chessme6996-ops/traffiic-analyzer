from fastapi import FastAPI
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# Enable CORS so your React app (Port 3000) can talk to this API (Port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
# Ensure 'traffic_model.pkl' is in the same folder as this script
try:
    model = pickle.load(open("traffic_model.pkl", "rb"))
except Exception as e:
    print(f"Error loading model: {e}")

@app.get("/predict/{junction_id}/{hour}")
def predict(junction_id: str, hour: int):
    try:
        file_path = f"junction_{junction_id.upper()}.csv"
        df = pd.read_csv(file_path)
        
        # Calculate averages from your dataset (1-39s)
        avg_row = df.mean(numeric_only=True)

        # Features: [hour, car, bike, bus, truck, total]
        features = [[
            hour, 
            avg_row.get("car", 0),
            avg_row.get("bike", 0),
            avg_row.get("bus", 0),
            avg_row.get("truck", 0),
            avg_row.get("total", 0)
        ]]

        # The model returns a label like "Medium"
        prediction = model.predict(features)[0]

        return {
            "congestion": str(prediction), # Pass the word "Medium" to React
            "expected_cars": int(avg_row.get("car", 0)),
            "expected_bikes": int(avg_row.get("bike", 0)),
            "expected_buses": int(avg_row.get("bus", 0)),
            "expected_trucks": int(avg_row.get("truck", 0))
        }
    except Exception as e:
        return {"error": str(e), "congestion": "Error"}

if __name__ == "__main__":
    import uvicorn
    # Start the server on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)