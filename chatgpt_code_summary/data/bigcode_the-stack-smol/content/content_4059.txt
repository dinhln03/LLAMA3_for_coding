import uvicorn
from fastapi import (FastAPI, File, UploadFile)
from starlette.responses import RedirectResponse
from tensorflow.python.keras.preprocessing import image as imgx
import requests
from PIL import Image
from application.components import predict, read_imagefile
from application.schema import Symptom
from application.components.prediction import symptom_check
from googletrans import Translator, constants
from pprint import pprint
app_desc = """<h2>Try this app by uploading any image with `predict/image`</h2>
<h2>Analize photos</h2>
<br>Template by Aniket Maurya, new version by Joaquin Egocheaga"""

app = FastAPI(title='Comparizy  ,  Tensorflow FastAPI ', description=app_desc)

translator = Translator()
@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    print(file.filename)
    print(extension)
    if not extension:
        return "Image must be jpg or png format!"

    image = read_imagefile(await file.read())
    
         
    prediction = predict(image)

    clase=prediction[0]['class']
    clase=clase.replace("_", " ")
    print(clase)
    
    print("X")
    translation = translator.translate(clase, "es") 
    translation=translation.text
    print(translation)
    
    return translation


@app.post("/api/covid-symptom-check")
def check_risk(symptom: Symptom):
    return symptom_check.get_risk_level(symptom)


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
