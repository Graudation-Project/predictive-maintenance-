from models.Preprocessing import Preprocessing
from models.ModelSelection import ModelSelection
import joblib
from fastapi import FastAPI

app = FastAPI()
p_data = None

@app.get("/insertData")
async def get_data(data):
    p_data = data
    return {"Done"}


@app.get("/predict")
async def get_data(data):
    prediction = model.predict(data)
    return {"Prediction":prediction}








if __name__ == "__main__":
    data = None

    pre = Preprocessing(p_data)
    cleaned_data = pre.clean_data()

    model_selection = ModelSelection(cleaned_data)
    model = model_selection.training()
    # Save Model
    joblib.dump(model,"model.joblib")

    
