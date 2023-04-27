import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.applications.vgg16 import VGG16



# Load the VGG16 model
model = VGG16(weights='imagenet')

# Load the two input images
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# Preprocess the input images
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image1 = cv2.resize(image1, (224, 224))
image1 = np.expand_dims(image1, axis=0)
image1 = preprocess_input(image1)

image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
image2 = cv2.resize(image2, (224, 224))
image2 = np.expand_dims(image2, axis=0)
image2 = preprocess_input(image2)

# Compute the output features for the two images
features1 = model.predict(image1)
features2 = model.predict(image2)

# Compute the absolute difference between the two feature vectors
diff = np.abs(features1 - features2)

# Compute the similarity score
similarity = np.sum(diff)

# Print the similarity score
print("Similarity score: ", similarity)