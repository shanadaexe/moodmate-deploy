from uuid import uuid4
import os
import numpy as np
import cv2
from keras.applications.vgg16 import preprocess_input
from keras.src.applications import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from PIL import Image

from keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import cv2
import time
import numpy as np
import time
from lime.lime_image import LimeImageExplainer

import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model, load_model

import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import Model, load_model
from lime.lime_image import LimeImageExplainer
from PIL import Image
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

folder_path = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/src/CaptureImages'

resNet50Model_predict = None
img_scaled_resNet50 = None
vgg50GradCam_img_original = None
vgg50GradCam_img_for_model = None
vgg50GradCam_model= None
target_size_resNet50 = (244, 244)
def predict_emotion_resNet50(image_path):
    global resNet50Model_predict
    global img_scaled_resNet50
    global vgg50GradCam_img_original
    global vgg50GradCam_img_for_model
    global vgg50GradCam_model
    model_path = '/Users/isurudissanayake/Documents/Data/DATA_SET/EMOTION/Feature_extraction/ResNet50/emotion_model.h5'
    target_size = (224, 224)

    # Load the pre-trained VGG16 model with top layers included
    base_model = ResNet50(weights='imagenet', include_top=True)

    gradCamModel = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    vgg50GradCam_model = gradCamModel

    # Take the output of the base model up to the last convolutional layer
    x = base_model.get_layer('avg_pool').output

    # Add a new dense layer for output
    x = Dense(2048, activation='relu')(x)
    x = Reshape((1, 1, 2048))(x)
    x = GlobalAveragePooling2D()(x)

    predictions = Dense(1, activation='sigmoid')(x)

    # Create a new model that takes the input of VGG16 and outputs the desired layer
    model = Model(inputs=base_model.input, outputs=predictions)


    vgg50GradCam_img_original = cv2.imread(image_path)
    vgg50GradCam_img_original = vgg50GradCam_img_original.copy()

    vgg50GradCam_img_original = cv2.resize(vgg50GradCam_img_original, target_size)
    vgg50GradCam_img_for_model = preprocess_input(np.expand_dims(vgg50GradCam_img_original, axis=0))

    # Process the input image
    img = np.array(Image.open(image_path).resize(target_size))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    img = preprocess_input(img)  # Preprocess the image

    # Preprocess the image for the explainer by dividing pixel values by 255
    img_scaled_resNet50 = img / 255.0

    # Predict ASD probability using the full VGG16 model
    resNet50Model_predict = model.predict
    prediction = model.predict(img)  # Access the first element for ASD probability

    emotion_labels = ['Angry', 'Fear', 'Joy', 'Natural', 'Sadness', 'Surprise']
    predicted_emotion_index = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_emotion_index]
    prediction_probability = prediction[0][predicted_emotion_index]

    print(f"Predicted emotion: {predicted_emotion}")
    print(f"Prediction probability: {prediction_probability:.2f}")

    return predicted_emotion, prediction_probability


def request_gradcam_emotion_resNet50():
    print("request_gradcam_emotion_resNet50")
    global vgg50GradCam_img_original
    global vgg50GradCam_img_for_model
    global vgg50GradCam_model

    heatmap = generate_grad_cam(vgg50GradCam_model, vgg50GradCam_img_for_model, 'conv5_block3_out')

    # Resize heatmap to match the size of the original image
    heatmap = cv2.resize(heatmap, (vgg50GradCam_img_original.shape[1], vgg50GradCam_img_original.shape[0]))

    # Apply colormap for better visualization
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(vgg50GradCam_img_original, 0.6, heatmap, 0.4, 0)

    # Save the Grad-CAM image to the specified location
    save_folder = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/public/Emotion_Prediction/GradCAM/GradCAM_ResNet50'

    unique_filename = str(uuid4()) + '.jpg'
    output_image_path = os.path.join(save_folder, unique_filename)

    cv2.imwrite(output_image_path, superimposed_img)
    print(f"Grad-CAM output saved at: {output_image_path}")

    returnOutput = './Emotion_Prediction/GradCAM/GradCAM_ResNet50/' + unique_filename
    print(returnOutput)
    return returnOutput


def request_lime_emotion_resNet50(image_path):
    global resNet50Model_predict
    global img_scaled_resNet50

    explainer = LimeImageExplainer()

    # Generate an explanation for the prediction using the explainer object
    explanation = explainer.explain_instance(img_scaled_resNet50[0], resNet50Model_predict, top_labels=1, hide_color=0,
                                             num_samples=10000, random_seed=42)

    # Visualize the explanation using matplotlib
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=False)

    # Resize the explanation mask to match the original image dimensions
    mask = cv2.resize(mask, (target_size_resNet50[0], target_size_resNet50[1]), interpolation=cv2.INTER_NEAREST)

    # Convert the mask to the original image mode
    original_image = Image.open(image_path)
    # original_image = original_image.convert("L")  # Convert the original image to grayscale
    original_width, original_height = original_image.size
    original_mode = original_image.mode

    # Overlay the explanation mask on the original image
    mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    original_image = np.array(original_image)
    original_image[mask > 0.5] = (0, 255, 0)

    # Save the modified image to the specified location
    save_folder = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/public/Emotion_Prediction/LIME/LIME_ResNet50'

    unique_filename = str(uuid4()) + '.jpg'
    output_image_path = os.path.join(save_folder, unique_filename)
    output_image = Image.fromarray(original_image)
    output_image.save(output_image_path)

    returnOutput = './Emotion_Prediction/LIME/LIME_ResNet50/' + unique_filename

    return returnOutput

def generate_grad_cam(model, img_array, layer_name):
    # Create a model that maps the input image to the desired layer's output
    grad_model = Model(inputs=model.input, outputs=(model.get_layer(layer_name).output, model.output))

    # Compute the gradient of the predicted class with respect to the output feature map of the given layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        predicted_class_output = preds[:, 0]  # Assuming ASD class is the first one

    grads = tape.gradient(predicted_class_output, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]

    # Compute the heatmap
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU on the heatmap
    heatmap /= np.max(heatmap)  # Normalize

    return heatmap