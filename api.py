from fastapi import FastAPI
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        with open("traffic_model.pkl", "rb") as f:
            model = pickle.load(f)
        print(" Model loaded ")
    except Exception as e:
        print(f" Error: {e}")

@app.get("/predict/{junction_id}/{hour}")
def predict(junction_id: str, hour: int):
    try:
        file_path = f"junction_{junction_id.upper()}.csv"
        df = pd.read_csv(file_path)
        
        
        
        hourly_data = df[df['time_sec'] == hour]

        if hourly_data.empty:
            
            avg_row = df.mean(numeric_only=True)
        else:
            avg_row = hourly_data.mean(numeric_only=True)

        
        feature_dict = {
            "hour": [hour],
            "car": [int(avg_row.get("car", 0))],
            "bike": [int(avg_row.get("bike", 0))],
            "bus": [int(avg_row.get("bus", 0))],
            "truck": [int(avg_row.get("truck", 0))],
            "total": [int(avg_row.get("total", 0))]
        }
        features_df = pd.DataFrame(feature_dict)
        
        if model is None:
            return {"error": "Model not loaded", "congestion": "Error"}

        prediction = model.predict(features_df)[0]

        return {
            "congestion": str(prediction), 
            "expected_cars": int(avg_row.get("car", 0)),
            "expected_bikes": int(avg_row.get("bike", 0)),
            "expected_buses": int(avg_row.get("bus", 0)),
            "expected_trucks": int(avg_row.get("truck", 0))
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"error": str(e), "congestion": "Error"}
            
        prediction = model.predict(features_df)[0]

       
        return {
            "congestion": str(prediction), 
            "expected_cars": int(avg_row.get("car", 0)),
            "expected_bikes": int(avg_row.get("bike", 0)),
            "expected_buses": int(avg_row.get("bus", 0)),
            "expected_trucks": int(avg_row.get("truck", 0))
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        return {"error": str(e), "congestion": "Error"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
