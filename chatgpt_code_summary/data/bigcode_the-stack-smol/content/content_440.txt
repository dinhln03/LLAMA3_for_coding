from io import BytesIO
import pytest
from app import app

def test_otter():
  with open('./images/otter.jpeg', 'rb') as img:
    img_string = BytesIO(img.read())
  response = app.test_client().post('/predict', data={'file': (img_string, 'otter.jpeg')},
                      content_type="multipart/form-data")
  assert response.json['class_name'] == 'otter'
