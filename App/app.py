import json
from flask import Flask, redirect, render_template, request, session, url_for
import os
from PIL import Image
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from flask_babel import Babel, _
import cv2
import matplotlib.cm as cm

# -----------------------------
# Load CSV Files
# -----------------------------
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# -----------------------------
# Load Model
# -----------------------------
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 39)
model.load_state_dict(torch.load("ResNet50.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Prediction Function
# -----------------------------
def predict(file_path=None):
    try:
        img = Image.open(file_path).convert("RGB")
        img = transform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            output = model(img)
            return torch.argmax(output, dim=1).item()

    except Exception as e:
        print("Prediction Error:", e)
        return None

# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__)
app.secret_key = "your_secret_key_here"

LANGUAGES = {"en": "English", "mr": "Marathi", "hi": "Hindi"}

app.config['LANGUAGES'] = LANGUAGES
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'

babel = Babel()

def get_locale():
    return session.get('lang', 'en')

babel.init_app(app, locale_selector=get_locale)

# -----------------------------
# Change Language Route
# -----------------------------
@app.route('/change_lang/<lang>')
def change_lang(lang):
    if lang in LANGUAGES:
        session['lang'] = lang
    return redirect(request.referrer or url_for('home_page'))

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

# -----------------------------
# SEVERITY MODEL
# -----------------------------
def calculate_severity(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return 0, "Unknown"

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([5, 30, 20])
    upper = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    infected_pixels = np.sum(mask > 0)
    total_pixels = mask.size

    severity_percent = (infected_pixels / total_pixels) * 100

    if severity_percent < 25:
        level = "Mild"
    elif severity_percent < 50:
        level = "Moderate"
    else:
        level = "Severe"

    return round(severity_percent, 2), level

# -----------------------------
# SUBMIT ROUTE
# -----------------------------
@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)

        pred = predict(file_path)

        severity_percent, severity_level = calculate_severity(file_path)

        heatmap_path = generate_heatmap(file_path)

        gradcam_path = generate_gradcam(file_path)

        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        early_care = disease_info['early_care'][pred]


        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        whatsapp_message = (
    f"*Crop Disease Alert!*\n"
    f"Detected Disease: {title}\n"
    f"Severity: {severity_level} ({severity_percent}%)\n\n"
    f"Description:\n{description}\n\n"
    f"Early Care:\n{early_care}\n\n"
    f"Prevention:\n{prevent}\n\n"
    f"Suggested Supplement: {supplement_name}\n"
    f"Buy here: {supplement_buy_link}"
    )

        from urllib.parse import quote

        whatsapp_text = quote(whatsapp_message)

        return render_template(
            'submit.html',
            title=title,
            desc=description,
            prevent=prevent,
            early_care=early_care, 
            image_url=image_url,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image_url,
            buy_link=supplement_buy_link,
            severity_percent=severity_percent,
            severity_level=severity_level,
            heatmap=heatmap_path,
            gradcam=gradcam_path,
            whatsapp_text=whatsapp_text


        )

# MARKET ROUTE
@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )

# ---------------------------------------------
# Generate Heatmap
# ---------------------------------------------
def generate_heatmap(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Error loading image for heatmap.")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([5, 30, 20])
    upper = np.array([30, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    blended = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    heatmap_filename = "heatmap_" + os.path.basename(image_path)
    heatmap_path = os.path.join("static/heatmaps", heatmap_filename)

    cv2.imwrite(heatmap_path, blended)

    return heatmap_path

# ---------------------------------------------
# Grad-CAM
# ---------------------------------------------
def generate_gradcam(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)

    # Forward & backward hook logic
    features = gradients = None

    def forward_hook(module, _, output):
        nonlocal features
        features = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    target_layer = model.layer4[2].conv3
    hook_f = target_layer.register_forward_hook(forward_hook)
    hook_b = target_layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    pred_class = output.argmax()
    
    model.zero_grad()
    output[0, pred_class].backward()

    hook_f.remove()
    hook_b.remove()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    cam = torch.zeros(features.shape[2:], dtype=torch.float32)

    for i, w in enumerate(pooled_gradients):
        cam += w * features[0, i, :, :]

    cam = torch.relu(cam)
    cam -= cam.min()
    cam /= cam.max()

    cam = cv2.resize(cam.cpu().detach().numpy(), (img.width, img.height))
    heatmap = cm.jet(cam)[:, :, :3] * 255
    heatmap = heatmap.astype("uint8")

    original = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    gradcam_filename = "gradcam_" + os.path.splitext(os.path.basename(image_path))[0] + ".jpg"
    gradcam_path = os.path.join("static/heatmaps", gradcam_filename)

    cv2.imwrite(gradcam_path, superimposed)
    print(f"Grad-CAM saved at: {gradcam_path}")

    return gradcam_path

# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
