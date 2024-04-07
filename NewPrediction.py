import numpy as np
from uuid import uuid4
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Reshape, Dense
from PIL import Image
from pathlib import Path
from IPython.display import display
from lime.lime_image import LimeImageExplainer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import time

# Load the trained model for emotion prediction
emotion_model = load_model('/Users/isurudissanayake/Desktop/FASD/ResNet50/emotion_model.h5')

# Load the VGG16 model without the top classification layers
resnet50 = ResNet50(weights='imagenet', include_top=False)

# Load the pre-trained VGG16 model for ASD classification
resnet50_model = load_model('/Users/isurudissanayake/Documents/Data/DATA_SET/Feature-Extraction/ResNet50/ResNet50Model.h5')

base_model = ResNet50(weights='imagenet', include_top=True)

x = base_model.get_layer('avg_pool').output
x = Dense(2048, activation='relu')(x)
x = Reshape((1, 1, 2048))(x) # Add this line to reshape the output of GlobalAveragePooling2D
x = GlobalAveragePooling2D()(x)
prediction = Dense(1, activation='sigmoid')(x)

# Create a new model that takes the input of ResNet50 and outputs the desired layer
resNet50Model = Model(inputs=base_model.input, outputs=prediction)

target_size = (224, 224)
resNet50_Img_scaled = None

resnet50_grad_img_original = None
resnet50_grad_img_for_model = None
resnet50_grad_img_scaled = None

def generate_grad_cam(model, img_array, layer_name):
    # Create a model that maps the input image to the desired layer's output
    grad_model = Model(inputs=model.input, outputs=(model.get_layer(layer_name).output, model.output))

    # Compute the gradient of the predicted class with respect to the output feature map of the given layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        predicted_class_output = preds[:, 0]  # ASD class index assuming ASD class is the first one

    grads = tape.gradient(predicted_class_output, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]

    # Compute the heatmap
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU on the heatmap
    heatmap /= np.max(heatmap)  # Normalize

    return heatmap

# Define a function to preprocess input image and extract features for emotion prediction
def preprocess_image(image_path, feature_extraction_model):
    global resNet50_Img_scaled
    global resnet50_grad_img_original
    global resnet50_grad_img_for_model
    global resnet50_grad_img_scaled

    gradImg = cv2.imread(image_path)
    resnet50_grad_img_original = gradImg.copy()  # Save a copy for visualization later
    gradImg = cv2.resize(gradImg, target_size)
    resnet50_grad_img_for_model = preprocess_input(np.expand_dims(image.img_to_array(gradImg), axis=0))

    resnet50_grad_img_scaled = gradImg / 255.0

    limeImg = cv2.imread(image_path)
    limeImg = cv2.resize(limeImg, target_size)
    limeImg = cv2.cvtColor(limeImg, cv2.COLOR_BGR2RGB)
    limeImg = np.expand_dims(limeImg, axis=0)
    limeImg = preprocess_input(limeImg)

    resNet50_Img_scaled = limeImg / 255.0

    # Load and preprocess the input image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extract features
    features = feature_extraction_model.predict(img_array)

    return features

# Define a function to predict emotion from an input image
def predictEmotion(image_path):
    print("PredictEmotion")
    # Preprocess the input image and extract features
    features = preprocess_image(image_path, resnet50)

    # Make prediction
    prediction = emotion_model.predict(features)

    # Decode the prediction
    emotions = ['Angry', 'Fear', 'Joy', 'Sad']
    predicted_emotion_index = np.argmax(prediction)
    predicted_emotion = emotions[predicted_emotion_index]

    # Get percentage of prediction for each emotion
    percentages = {emotion: round(float(prediction[0][i]) * 100, 2) for i, emotion in enumerate(emotions)}

    # Find the emotion with the maximum percentage
    max_emotion = max(percentages, key=percentages.get)
    max_percentage = percentages[max_emotion]

    print(f"Predicted emotion: {max_emotion}")
    print(f"Prediction probability: {max_percentage:.2f}%")

    return max_emotion, max_percentage



def request_GradCAM_emotion():
    global resNet50Model
    global resnet50_grad_img_original
    global resnet50_grad_img_for_model
    global resnet50_grad_img_scaled

    # Generate class activation heatmap
    heatmap = generate_grad_cam(resNet50Model, resnet50_grad_img_for_model, 'conv5_block3_out')

    # Resize the heatmap to match the original image
    heatmap = cv2.resize(heatmap, (resnet50_grad_img_original.shape[1], resnet50_grad_img_original.shape[0]))

    # Convert the heatmap to the RGB color map
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(resnet50_grad_img_original, 0.6, heatmap, 0.4, 0)

    # Save the modified image to the specified location
    save_folder = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/public/Emotion_Prediction/GradCAM/GradCAM_ResNet50'

    unique_filename = str(uuid4()) + '.jpg'
    output_image_path = os.path.join(save_folder, unique_filename)

    cv2.imwrite(output_image_path, superimposed_img)
    print(f"Grad-CAM output saved at: {output_image_path}")

    returnOutput = './Emotion_Prediction/GradCAM/GradCAM_ResNet50/' + unique_filename

    return returnOutput


def request_lime_emotion(image_path):
    global resNet50Model
    global resNet50_Img_scaled

    explainer = LimeImageExplainer()

    # Generate an explanation for the prediction using the explainer object
    explanation = explainer.explain_instance(resNet50_Img_scaled[0], resNet50Model, top_labels=1, hide_color=0,
                                             num_samples=10000, random_seed=42)

    # Visualize the explanation using matplotlib
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=False)

    # Resize the explanation mask to match the original image dimensions
    mask = cv2.resize(mask, (target_size[0], target_size[1]), interpolation=cv2.INTER_NEAREST)

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


def predict_ASD(image_path):
    global resnet50_model
    global resnet50

    global resNet50_Img_scaled

    global resnet50_grad_img_original
    global resnet50_grad_img_for_model
    global resnet50_grad_img_scaled
    # Preprocess the input image and extract features
    features = preprocess_image(image_path, resnet50)

    # Make prediction
    prediction = resnet50_model.predict(features)

    # Decode the prediction
    if prediction > 0.5:
        # explainer = LimeImageExplainer()
        #
        # # Generate an explanation for the prediction using the explainer object
        # explanation = explainer.explain_instance(resNet50_Img_scaled[0], resNet50Model.predict, top_labels=1, hide_color=0, num_samples=10000, random_seed=42)
        #
        # # Visualize the explanation using matplotlib
        # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
        #
        # # Resize the explanation mask to match the original image dimensions
        # mask = cv2.resize(mask, (target_size[0], target_size[1]), interpolation=cv2.INTER_NEAREST)
        #
        # # Convert the mask to the original image mode
        # original_image = Image.open(input_image_path)
        # #original_image = original_image.convert("L")  # Convert the original image to grayscale
        # original_width, original_height = original_image.size
        # original_mode = original_image.mode
        #
        # # Overlay the explanation mask on the original image
        # mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        # original_image = np.array(original_image)
        # original_image[mask > 0.5] = (0, 255, 0)
        #
        # # Display the original image with the explanation mask
        # display(Image.fromarray(original_image))

        # # Visualize the Grad-CAM heatmap
        # heatmap = generate_grad_cam(resNet50Model, resnet50_grad_img_for_model, 'conv5_block3_out')
        #
        # # Resize heatmap to match the size of the original image
        # heatmap = cv2.resize(heatmap, (resnet50_grad_img_original.shape[1], resnet50_grad_img_original.shape[0]))
        #
        # # Apply colormap for better visualization
        # heatmap = np.uint8(255 * heatmap)
        # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #
        # # Superimpose the heatmap on the original image
        # superimposed_img = cv2.addWeighted(resnet50_grad_img_original, 0.6, heatmap, 0.4, 0)
        #
        # # Display the superimposed image
        # plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()
        print(f"ASD Prediction: {round(float(prediction[0]) * 100, 2)}%")

        return round(float(prediction[0]) * 100, 2)
    else:
        print(f"Non-ASD Prediction: {round(float(1 - prediction[0]) * 100, 2)}%")

        return round(float(prediction[0]) * 100, 2)
        # return (float(1 - prediction[0]) * 100, 2)


def request_GradCAM_ASD():
    global resNet50Model
    global resnet50_grad_img_original
    global resnet50_grad_img_for_model
    global resnet50_grad_img_scaled

    # Generate class activation heatmap
    heatmap = generate_grad_cam(resNet50Model, resnet50_grad_img_for_model, 'conv5_block3_out')

    # Resize the heatmap to match the original image
    heatmap = cv2.resize(heatmap, (resnet50_grad_img_original.shape[1], resnet50_grad_img_original.shape[0]))

    # Convert the heatmap to the RGB color map
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(resnet50_grad_img_original, 0.6, heatmap, 0.4, 0)

    # Save the modified image to the specified location
    save_folder = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/public/ASD_Prediction/GradCAM/GradCAM_ResNet50'

    unique_filename = str(uuid4()) + '.jpg'
    output_image_path = os.path.join(save_folder, unique_filename)

    cv2.imwrite(output_image_path, superimposed_img)
    print(f"Grad-CAM output saved at: {output_image_path}")

    returnOutput = './ASD_Prediction/GradCAM/GradCAM_ResNet50/' + unique_filename

    return returnOutput


def request_Lime_ASD(image_path):
    global resNet50Model
    global resNet50_Img_scaled

    explainer = LimeImageExplainer()

    # Generate an explanation for the prediction using the explainer object
    explanation = explainer.explain_instance(resNet50_Img_scaled[0], resNet50Model.predict, top_labels=1, hide_color=0, num_samples=10000, random_seed=42)

    # Visualize the explanation using matplotlib
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)

    # Resize the explanation mask to match the original image dimensions
    mask = cv2.resize(mask, (target_size[0], target_size[1]), interpolation=cv2.INTER_NEAREST)

    # Convert the mask to the original image mode
    original_image = Image.open(image_path)
    #original_image = original_image.convert("L")  # Convert the original image to grayscale
    original_width, original_height = original_image.size
    original_mode = original_image.mode

    # Overlay the explanation mask on the original image
    mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    original_image = np.array(original_image)
    original_image[mask > 0.5] = (0, 255, 0)

    # Save the modified image to the specified location
    save_folder = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/public/ASD_Prediction/LIME/LIME_ResNet50'

    unique_filename = str(uuid4()) + '.jpg'
    output_image_path = os.path.join(save_folder, unique_filename)
    output_image = Image.fromarray(original_image)
    output_image.save(output_image_path)

    returnOutput = './ASD_Prediction/LIME/LIME_ResNet50/' + unique_filename

    return returnOutput

# # Path to the input image
# input_image_path = '/Users/isurudissanayake/Documents/Data/DATA_SET/ASD/0049.jpg'
#
# # Predict emotion of the input image
# predicted_emotion, percentages = predict_emotion(input_image_path, emotion_model, resnet50)
# print("Predicted Emotion:", predicted_emotion)
# print("Prediction Percentages:", percentages)
#
# # Predict ASD of the input image
# predicted_asd_label, asd_percentage = predict_asd(input_image_path, resnet50_model, resnet50)
# print("Predicted ASD Label:", predicted_asd_label)
# print("ASD Percentage:", asd_percentage)