import os
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Fixture to upload a sample image file
@pytest.fixture
def sample_image():
    return open("/Users/isurudissanayake/Documents/Data/DATA_SET/ASD/0049.jpg", "rb")

def test_upload_image(sample_image):
    response = client.post("/upload_image", files={"image": sample_image})
    assert response.status_code == 200
    assert "filename" in response.json()

def test_save_image(sample_image):
    response = client.post("/save_image", files={"image": sample_image})
    assert response.status_code == 200
    assert "filename" in response.json()

def test_predict_asd():
    response = client.get("/predictASD?filepath=sample_image.jpg")
    assert response.status_code == 200
    assert 'error' in response.json()
    assert response.json()['error'] == "Failed to predict ASD: 'NoneType' object has no attribute 'copy'"


def test_predict_emotion():
    response = client.get("/predictEmotoin?filepath=sample_image.jpg")
    assert response.status_code == 200
    assert 'error' in response.json()
    assert response.json()['error'] == "Failed to predict Emotion: 'NoneType' object has no attribute 'copy'"

def test_emotion_explain_lime():
    response = client.get("/emotion-xai-lime?filepath=sample_image.jpg")
    assert response.status_code == 200
    assert 'error' in response.json()
    assert response.json()['error'] == "Failed to explain LIME: 'NoneType' object is not subscriptable"

def test_emotion_explain_gradcam():
    response = client.get("/emotion-xai-gradcam?filepath=sample_image.jpg")
    assert response.status_code == 200
    assert 'error' in response.json()
    assert response.json()['error'] == "Failed to explain GradCAM: Inputs to a layer should be tensors. Got 'None' (of type <class 'NoneType'>) as input for layer 'model'."

def test_xai_lime():
    response = client.get("/xai-lime?filepath=sample_image.jpg")
    assert response.status_code == 200
    assert 'error' in response.json()
    assert response.json()['error'] == "Failed to explain LIME: 'NoneType' object is not subscriptable"

def test_xai_gradcam():
    response = client.get("/xai-gradcam?filepath=sample_image.jpg")
    assert response.status_code == 200
    assert 'error' in response.json()
    assert response.json()['error'] == "Failed to explain GradCAM: Inputs to a layer should be tensors. Got 'None' (of type <class 'NoneType'>) as input for layer 'model'."

# Clean up generated image files after tests
def teardown_function():
    for file in os.listdir("/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/src/CaptureImages"):
        if file.endswith(".jpg"):
            os.remove(os.path.join("/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/src/CaptureImages", file))

