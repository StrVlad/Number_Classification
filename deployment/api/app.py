from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from PIL import Image
import io
import torchvision.transforms as transforms

# Your exact model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load model
def load_model():
    model = SimpleNN()
    model.load_state_dict(torch.load('/app/models/mnist_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Same transformations as during training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

app = FastAPI(title="MNIST Classifier")

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "MNIST Classifier API - Your trained model!"}

@app.get("/model-info")
def model_info():
    return {
        "model": "SimpleNN",
        "layers": "128-64-10", 
        "accuracy": "~97-98%",
        "description": "Your model trained for 5 epochs"
    }

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        # Read and process image
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data)).convert('L')  # Grayscale
        
        # Resize and transform as during training
        img = img.resize((28, 28))
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        
        # Prediction
        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.argmax(output, 1).item()
            probabilities = torch.softmax(output, 1).numpy()[0]
        
        # Format probabilities for display
        prob_dict = {str(i): float(prob) for i, prob in enumerate(probabilities)}
        
        return {
            "success": True,
            "prediction": int(prediction),
            "probabilities": prob_dict,
            "confidence": float(probabilities[prediction])
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)