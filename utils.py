import numpy as np
from PIL import Image
import tifffile as tiff
from tensorflow.keras.models import load_model

# Load model
model = load_model('/Users/seifeldenelmizayen/Desktop/Cellula_Computer Vision/Fifth week/teeth_vgg16_model.h5')
#Channels to keep
channels = [1,2,3,4,5,6,8,12]

def compute_ndwi(img, green_channel = 2, nir_channel = 4):
    green = img[:, :, green_channel]
    nir = img[:, :, nir_channel]
    ndwi = (green - nir) / (green + nir + 1e-5)
    return ndwi

def preprocess_image(filepath):

    img = tiff.imread(filepath)

    # Transpose if shape is (12, H, W)
    if img.ndim == 3 and img.shape[0] == 12:
        img = np.transpose(img, (1, 2, 0))  # (H, W, 12)

    # Compute NDWI
    ndwi_map = compute_ndwi(img)
    ndwi_map = ndwi_map[..., np.newaxis]  # shape: (H, W, 1)

    # Concatenate NDWI as 13th channel
    img = np.concatenate([img, ndwi_map], axis=-1)  # shape: (H, W, 13)

    # Normalize each channel independently
    img = img.astype(np.float32)
    img = (img - np.min(img, axis=(0, 1))) / (np.max(img, axis=(0, 1)) - np.min(img, axis=(0, 1)) + 1e-5)

    img_selected = img[ :, :, channels]

    # Add batch dimension → shape: (1, 128, 128, 8)
    img_final = np.expand_dims(img_selected, axis=0)

    return img_final

def predict_mask(filepath):
    img = preprocess_image(filepath)
    mask = model.predict(img)[0]  # shape: (128, 128, 1)
    mask = np.squeeze(mask)       # shape: (128, 128)

    # Optional: Binarize mask (useful if model outputs probabilities)
    mask = (mask > 0.5).astype(np.uint8)

    return mask
