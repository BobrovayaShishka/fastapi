import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    logger.info("Модель и препроцессоры успешно загружены")
except Exception as e:
    logger.error(f"Ошибка загрузки: {e}")
    raise

app = FastAPI(title="House Price Predictor", version="1.0")


class HouseInput(BaseModel):
    district: str
    rooms: int
    square: float
    floor: int
    total_floors: int
    building_type: str
    condition: str


@app.post("/predict")
async def predict(house: HouseInput):
    try:
        input_data = pd.DataFrame([house.dict()])

        input_data['room_size'] = input_data['square'] / input_data['rooms']
        input_data['floor_ratio'] = input_data['floor'] / input_data['total_floors']
        input_data['is_top_floor'] = (input_data['floor'] == input_data['total_floors']).astype(int)

        for col in ["district", "building_type", "condition"]:
            input_data[col] = input_data[col].astype('category').cat.codes

        expected_columns = [
            'district', 'rooms', 'square', 'floor', 'total_floors',
            'building_type', 'condition', 'room_size', 'floor_ratio', 'is_top_floor'
        ]
        input_data = input_data[expected_columns]

        scaled_data = scaler.transform(input_data)

        assert (input_data.columns == model.feature_names_in_).all(), \
            f"Columns mismatch!\nModel expects: {model.feature_names_in_}\nGot: {input_data.columns.tolist()}"

        prediction = model.predict(scaled_data)[0]

        return {"predicted_price": round(float(prediction), 2), "currency": "USD"}

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}


@app.get("/health")
async def health_check():
    return {"status": "OK", "service": "House Price Prediction"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
        workers=1,
        loop="asyncio",
        reload=False
    )