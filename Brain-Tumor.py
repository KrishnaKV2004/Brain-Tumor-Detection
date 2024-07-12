import streamlit as st
import tensorflow as tf
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def model_prediction(test_image):
    model = tf.keras.models.load_model('Brain_Tumor_Model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size = (128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    
    return result_index

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(" ", ["Home", "About", "Brain Tumor Detection"])

if (app_mode=="Home"):
    st.header("Brain Tumor Detection")
    st.text(" ")
    st.markdown(
    """ 
    Welcome to the Brain Tumor Detection System! 🌿🔍
    
    Our mission is to help in identifying brain tumor efficiently.
    Upload an MRI image, and our system will analyze it to detect any signs of diseases.
    Together.

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload MRI image with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Brain Tumor Detection** page in the sidebar to upload an image and experience the power of our Brain Tumor Recognition System!
    """
    )
    
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. Train (70295 images)
                2. Test (33 images)
                3. Validation (17572 images)

                """)
    
else:
    st.header("Brain Tumor Detection")
    test_image = st.file_uploader("Choose An Image")
    
    if (st.button("Show Image")):
        st.image(test_image, use_column_width=True)
        
    if (st.button("Predict")):
        st.write("Prediction")
        result_index = model_prediction(test_image)
        class_name = ['Glioma', 'Meningioma', 'Normal', 'Pituitary']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))