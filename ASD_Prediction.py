from uuid import uuid4
import os
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import Model, load_model
from lime.lime_image import LimeImageExplainer
from PIL import Image
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input


resNet50Model_predict = None
img_scaled_ResNet50 = None
resNet50GradCam_img_original = None
resNet50GradCam_img_for_model = None
resNet50GradCam_model = None
target_size_ResNet50 = (244, 244)
def predict_asd_ResNet50(image_path):
    global resNet50Model_predict
    global img_scaled_ResNet50
    global resNet50GradCam_img_original
    global resNet50GradCam_img_for_model
    global resNet50GradCam_model

    model_path = '/Users/isurudissanayake/Documents/Data/DATA_SET/Feature-Extraction/ResNet50/ResNet50Model.h5'
    target_size = (224, 224)

    base_model = ResNet50(weights='imagenet', include_top=True)
    x = base_model.get_layer('avg_pool').output

    x = Dense(255, activation='relu')(x)
    x = Reshape((1, 1, 255))(x)  # Add this line to reshape the output of GlobalAveragePooling2D
    x = GlobalAveragePooling2D()(x)
    prediction = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=prediction)
    resNet50GradCam_model = model

    img = cv2.imread(image_path)
    resNet50GradCam_img_original = img.copy()
    img = cv2.resize(img, target_size)
    resNet50GradCam_img_for_model = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    img_scaled_ResNet50 = img / 255

    resNet50Model_predict = model.predict
    prediction = model.predict(img)[0][0]  # Access the first element for ASD probability
    print("prediction: {:.5f}".format(prediction))

    rounded_prediction = round(prediction, 2)
    print(f"Predicted probability: {rounded_prediction:.2f}")

    if rounded_prediction > 0.5:
        print(f"Predicted ASD with probability: {rounded_prediction:.2f}")

        return rounded_prediction
    else:
        print(f"Predicted non-ASD with probability: {1 - rounded_prediction:.2f}")
        return rounded_prediction


def request_GradCam_ResNet50():
    print("request_GradCam_ResNet50")
    global resNet50GradCam_img_original
    global resNet50GradCam_img_for_model
    global resNet50GradCam_model

    # Generate class activation heatmap
    heatmap = generate_grad_cam(resNet50GradCam_model, resNet50GradCam_img_for_model, 'conv5_block3_out')

    # Resize the heatmap to match the original image
    heatmap = cv2.resize(heatmap, (resNet50GradCam_img_original.shape[1], resNet50GradCam_img_original.shape[0]))

    # Convert the heatmap to the RGB color map
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(resNet50GradCam_img_original, 0.6, heatmap, 0.4, 0)

    # Save the modified image to the specified location
    save_folder = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/public/ASD_Prediction/GradCAM/GradCAM_ResNet50'

    unique_filename = str(uuid4()) + '.jpg'
    output_image_path = os.path.join(save_folder, unique_filename)

    cv2.imwrite(output_image_path, superimposed_img)
    print(f"Grad-CAM output saved at: {output_image_path}")

    # output_image = Image.fromarray(superimposed_img)
    # output_image.save(output_image_path)

    returnOutput = './ASD_Prediction/GradCAM/GradCAM_ResNet50/' + unique_filename

    return returnOutput

def request_Lime_ResNet50(image_path):
    global resNet50Model_predict
    global img_scaled_ResNet50

    explainer = LimeImageExplainer()

    # Generate an explanation for the prediction using the explainer object
    explanation = explainer.explain_instance(img_scaled_ResNet50[0], resNet50Model_predict, top_labels=1, hide_color=0,
                                             num_samples=10000, random_seed=42)

    # Visualize the explanation using matplotlib
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                hide_rest=False)

    # Resize the explanation mask to match the original image dimensions
    mask = cv2.resize(mask, (target_size_ResNet50[0], target_size_ResNet50[1]), interpolation=cv2.INTER_NEAREST)

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
    save_folder = '/Users/isurudissanayake/DataspellProjects/FYP_Implementation/aunite/public/ASD_Prediction/LIME/LIME_ResNet50'

    unique_filename = str(uuid4()) + '.jpg'
    output_image_path = os.path.join(save_folder, unique_filename)
    output_image = Image.fromarray(original_image)
    output_image.save(output_image_path)

    returnOutput = './ASD_Prediction/LIME/LIME_ResNet50/' + unique_filename

    return returnOutput


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