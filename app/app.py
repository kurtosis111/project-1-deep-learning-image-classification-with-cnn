import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, render_template, jsonify, send_from_directory
import os

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'static/uploads' ##
os.makedirs(UPLOAD_FOLDER, exist_ok=True) ##

# Load model and class names
checkpoint = torch.load("model_inception.pth", map_location=torch.device("cpu"), weights_only=False)
class_names = checkpoint['class name']

model = models.inception_v3(weights="IMAGENET1K_V1")
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomVerticalFlip(),  # Random vertical flip
    transforms.ColorJitter(brightness=0.5, contrast=0.5),  # Random color adjustments
    transforms.Resize([299,299]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]),  ## is normalization really needed?
])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist('images')
    results = []

    for file in files:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        image = Image.open(filepath).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            prob, predicted = torch.max(probabilities, 1)
            predicted_class = class_names[predicted.item()]
            probability = round(prob.item() * 100, 4)

        # Infer actual class from filename (optional)
        actual_class = filename.split('_')[0]

        results.append({
            'image_url': f'/static/uploads/{filename}',
            'actual_class': actual_class,
            'predicted_class': predicted_class,
            'probability': probability
        })

        print(results)

    return render_template('results.html', results=results)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=8080)