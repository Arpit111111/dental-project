import streamlit as st
import numpy as np
import nibabel as nib
import plotly.graph_objs as go
import tensorflow as tf
from PIL import Image
import os

# Load the trained model
model = tf.keras.models.load_model('best_model.keras')

# Function to load CT image and convert to numpy array
def load_ct_image(image_path):
    img = nib.load(image_path).get_fdata()
    return img

# Function to create a 3D scatter plot from the CT data
def create_3d_scatter_plot(data):
    x, y, z = np.indices(data.shape)
    points = data > 0  # Threshold to create points
    x_points = x[points]
    y_points = y[points]
    z_points = z[points]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x_points.flatten(),
        y=y_points.flatten(),
        z=z_points.flatten(),
        mode='markers',
        marker=dict(size=1, color='blue', opacity=0.8)
    )])
    
    fig.update_layout(scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')),
        title='3D CT Scan Visualization',
        margin=dict(l=0, r=0, b=0, t=40))
    
    return fig

# Function to preprocess the CT scan for model prediction
def preprocess_ct_image(data):
    mid_slice = data[:, :, data.shape[2] // 2]  # Take the middle slice
    mid_slice = np.stack([mid_slice] * 3, axis=-1)  # Convert to 3 channels
    mid_slice = np.resize(mid_slice, (1, 112, 112, 3))  # Resize for the model
    return mid_slice.astype(np.float32) / 255.0  # Normalize

# Function to load and preprocess X-ray images
def load_xray_image(image_file):
    img = Image.open(image_file).convert("RGB")  # Ensure it's RGB
    img = img.resize((112, 112))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Streamlit UI
st.title("3D CT Scan & X-ray Visualization and Implant Prediction")

# File uploader for CT images
ct_file = st.file_uploader("Upload CT Scan (NIfTI .nii or .nii.gz)", type=["nii", "nii.gz"])
# File uploader for X-ray images
xray_file = st.file_uploader("Upload X-ray Image (JPEG, PNG)", type=["jpg", "jpeg", "png"])

if ct_file:
    ct_image_path = "ct_image.nii.gz"  # Temporary file path

    # Save the uploaded CT file to a temporary location
    with open(ct_image_path, "wb") as f:
        f.write(ct_file.getbuffer())
    
    try:
        with st.spinner("Loading CT scan..."):
            # Load CT image
            ct_data = load_ct_image(ct_image_path)
            
            # Create a 3D scatter plot
            fig = create_3d_scatter_plot(ct_data)
            st.plotly_chart(fig, use_container_width=True)

            # Preprocess CT scan for prediction
            processed_image = preprocess_ct_image(ct_data)
            prediction = model.predict(processed_image)

            # Display predicted coordinates
            st.write(f"Predicted Implant Coordinates from CT: X: {prediction[0][0]:.2f}, Y: {prediction[0][1]:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
    finally:
        # Cleanup temporary files
        if os.path.exists(ct_image_path):
            os.remove(ct_image_path)

if xray_file:
    st.image(xray_file, caption="Uploaded X-ray Image", use_column_width=True)

    # Load and preprocess the X-ray image
    xray_image = load_xray_image(xray_file)

    # Predict using the model
    xray_prediction = model.predict(xray_image)

    # Display predicted coordinates
    st.write(f"Predicted Implant Coordinates from X-ray: X: {xray_prediction[0][0]:.2f}, Y: {xray_prediction[0][1]:.2f}")

# Instructions for use
st.write("Upload a CT scan in NIfTI format and/or an X-ray image in JPEG or PNG format to visualize them in 3D and get implant placement predictions.")
