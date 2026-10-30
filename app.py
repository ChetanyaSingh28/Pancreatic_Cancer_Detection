import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Define custom objects required to load the saved model
def dice_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + 1e-7) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-7)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)

def iou_coef(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-7) / (union + 1e-7)

# Cache the model to prevent reloading on every UI interaction
@st.cache_resource
def load_unet_model():
    return tf.keras.models.load_model(
        "pancreas_unet.h5",
        custom_objects={
            "dice_coef": dice_coef,
            "dice_loss": dice_loss,
            "combined_loss": combined_loss,
            "iou_coef": iou_coef
        }
    )

st.title("Pancreas CT Segmentation Tool")
st.write("Upload a raw CT scan slice (.npy) to predict the pancreas mask.")

model = load_unet_model()
uploaded_file = st.file_uploader("Upload a CT Slice (.npy)", type=["npy"])

if uploaded_file is not None:
    # Load the uploaded numpy array
    ct_slice = np.load(uploaded_file)
    
    # Preprocess the slice 
    img = ct_slice.astype(np.float32)
    min_val, max_val = img.min(), img.max()
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)
        
    img_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    
    # Add channel and batch dimensions: (1, 128, 128, 1)
    model_input = np.expand_dims(np.expand_dims(img_resized, axis=-1), axis=0)

    # Generate prediction
    prediction = model.predict(model_input)
    pred_mask = (prediction[0] > 0.5).astype(np.uint8)

    # Visualize results
    st.subheader("Results")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img_resized, cmap="gray")
    axes[0].set_title("Input CT Scan")
    axes[0].axis("off")
    
    axes[1].imshow(pred_mask.squeeze(), cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")
    
    st.pyplot(fig)