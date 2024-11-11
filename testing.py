import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Set the paths
test_data_non_covid_dir = r"C:\Users\HP\Desktop\ML\test_data"  # Path to non-COVID images
test_data_covid_dir = r"C:\Users\HP\Desktop\ML\test_data_2"  # Path to COVID-infected images
model_path = r"C:\Users\HP\Desktop\ML\covid_model.h5"  # Path to the saved model file
image_size = (150, 150)  # Same size as used during training

# Load the trained model from the saved file
model = load_model(model_path)

# Function to preprocess a single image
def preprocess_image(image_path, image_size):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, image_size)
        img = img / 255.0  # Normalize pixel values to [0, 1]
        img = np.reshape(img, (1, image_size[0], image_size[1], 1))  # Add batch and channel dimensions
        return img
    else:
        print(f"Error: Unable to load image at {image_path}")
        return None

# Function to predict if an image has COVID or not
def predict_covid(model, image_path):
    img = preprocess_image(image_path, image_size)
    if img is not None:
        prediction = model.predict(img)
        predicted_label = int(np.round(prediction[0][0]))  # Get the binary prediction (0 or 1)
        return predicted_label  # Return the label (0 or 1) for counting later
    else:
        print(f"Skipping image {os.path.basename(image_path)} due to load error.")
        return None  # Return None if there's an issue with the image

# Counters for total images and prediction results
total_images = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0

# Function to process and count predictions for a given folder and label
def process_folder(folder_path, actual_label):
    global total_images, true_positives, false_positives, true_negatives, false_negatives
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(('.jpg', '.png', '.jpeg')):
            total_images += 1
            print(f"\nProcessing image: {file_name}")
            result = predict_covid(model, file_path)
            
            if result is not None:
                if result == 1:
                    if actual_label == 1:
                        true_positives += 1
                        print(f"Prediction: COVID detected in {file_name} (Correct)")
                    else:
                        false_positives += 1
                        print(f"Prediction: COVID detected in {file_name} (Incorrect)")
                else:
                    if actual_label == 0:
                        true_negatives += 1
                        print(f"Prediction: No COVID detected in {file_name} (Correct)")
                    else:
                        false_negatives += 1
                        print(f"Prediction: No COVID detected in {file_name} (Incorrect)")

# Process non-COVID and COVID folders
process_folder(test_data_non_covid_dir, actual_label=0)  # Non-COVID images
process_folder(test_data_covid_dir, actual_label=1)      # COVID-infected images

# Display the summary of predictions
print(f"\nTotal number of images: {total_images}")
print(f"True Positives (COVID detected correctly): {true_positives}")
print(f"False Positives (COVID incorrectly detected): {false_positives}")
print(f"True Negatives (Non-COVID detected correctly): {true_negatives}")
print(f"False Negatives (Non-COVID incorrectly detected): {false_negatives}")

# Calculate accuracy and precision
accuracy = ((true_positives + true_negatives) / total_images) * 100 if total_images > 0 else 0
precision = (true_positives / (true_positives + false_positives) * 100) if (true_positives + false_positives) > 0 else 0

# Display accuracy and precision
print(f"\nAccuracy: {accuracy:.2f}%")
print(f"Precision for detecting COVID: {precision:.2f}%")
