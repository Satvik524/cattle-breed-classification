from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from api.user_input import validate_image
from api.inference import predict_breed
from api.schemas import PredictionResponse
from api.model_loader import load_yolo_model, load_cattle_model

app = FastAPI()

cattle_model = load_cattle_model()
yolo_model = load_yolo_model()

@app.get('/')
def home():
    return {'message':'Welcome to the cattle prediction API'}

@app.get('/health')
def health():
    return {
        "status": "OK",
        "cattle_model_loaded": cattle_model is not None,
        "yolo_model_loaded": yolo_model is not None
    }

@app.post('/predict',response_model=PredictionResponse)
def predict_cattle(image: UploadFile = File(...)):
    validate_image(image)

    try:
        pil_image = Image.open(image.file)

        prediction = predict_breed(pil_image , yolo_model, cattle_model)

        if prediction is None:
            raise HTTPException(status_code = 422, detail='No cattle detected in the uploaded image')

        return prediction

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(status_code=500 , detail = 'An Error occured during the preprocessing')