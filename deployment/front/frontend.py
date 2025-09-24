import streamlit as st
import requests
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Digit Recognition", page_icon="🔢")

st.title("🔢 MNIST Digit Recognition")
st.markdown("**Using your trained model!**")

# Backend URL
BACKEND_URL = "http://backend:8000"

# Model information
try:
    response = requests.get(f"{BACKEND_URL}/model-info")
    if response.status_code == 200:
        info = response.json()
        st.sidebar.success(f"✅ Model loaded: {info['model']}")
        st.sidebar.write(f"Architecture: {info['layers']}")
        st.sidebar.write(f"Accuracy: {info['accuracy']}")
        st.sidebar.write(f"Description: {info['description']}")
except:
    st.sidebar.error("❌ Backend not available")

st.write("### Upload a digit image")
st.write("Works best with 28x28 pixel images, white digit on black background")

# Image upload
uploaded_file = st.file_uploader("Choose PNG, JPG, or JPEG file", 
                                type=['png', 'jpg', 'jpeg'])

col1, col2 = st.columns(2)

with col1:
    if uploaded_file is not None:
        # Show original image
        original_image = Image.open(uploaded_file).convert('L')
        st.image(original_image, caption="Original Image", use_container_width=True)
        
        # Show processed image (28x28)
        processed_image = original_image.resize((28, 28))
        
        # Create figure to display both images
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        
        ax[0].imshow(original_image, cmap='gray')
        ax[0].set_title('Original')
        ax[0].axis('off')
        
        ax[1].imshow(processed_image, cmap='gray')
        ax[1].set_title('Processed (28x28)')
        ax[1].axis('off')
        
        st.pyplot(fig)

with col2:
    if uploaded_file is not None:
        if st.button("🎯 Recognize Digit", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    # Send file to backend
                    files = {"image": uploaded_file.getvalue()}
                    response = requests.post(f"{BACKEND_URL}/predict/", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result["success"]:
                            prediction = result["prediction"]
                            confidence = result["confidence"]
                            probabilities = result["probabilities"]
                            
                            # Display result
                            st.success(f"### Result: digit **{prediction}**")
                            st.metric("Confidence", f"{confidence:.1%}")
                            
                            # Probability chart
                            st.write("### Probabilities for each digit:")
                            
                            digits = list(probabilities.keys())
                            probs = list(probabilities.values())
                            
                            # Create bar chart
                            fig, ax = plt.subplots(figsize=(10, 4))
                            bars = ax.bar(digits, probs, color='skyblue')
                            
                            # Highlight predicted digit
                            bars[prediction].set_color('red')
                            
                            ax.set_xlabel('Digit')
                            ax.set_ylabel('Probability')
                            ax.set_title('Probability Distribution')
                            ax.set_ylim(0, 1)
                            
                            # Add values on bars
                            for i, (digit, prob) in enumerate(zip(digits, probs)):
                                ax.text(i, prob + 0.01, f'{prob:.1%}', 
                                       ha='center', va='bottom')
                            
                            st.pyplot(fig)
                            
                            # Detailed probabilities table
                            st.write("### Detailed probabilities:")
                            for digit, prob in probabilities.items():
                                col1, col2, col3 = st.columns([1, 3, 1])
                                with col1:
                                    st.write(f"**{digit}**")
                                with col2:
                                    st.progress(prob)
                                with col3:
                                    st.write(f"{prob:.1%}")
                                    
                        else:
                            st.error(f"Error: {result.get('error', 'Unknown error')}")
                    else:
                        st.error("Connection error with server")
                        
                except Exception as e:
                    st.error(f"Error: {e}")

# Tips section
st.markdown("---")
st.write("### 💡 Tips for better recognition:")
st.write("""
- Use images with white digits on black background
- The digit should occupy most of the image
- Ideal size: 28x28 pixels
- Avoid blurry or rotated digits
""")