# CropCare – Plant Leaf Disease Prediction

CropCare is an AI/ML-powered web application that predicts 39 different plant leaf diseases using a trained ResNet-50 model and provides remedies + supplement recommendations. It includes multilingual support (English, Hindi, Marathi) using Flask-Babel.

## Features

1. Plant Leaf Disease Prediction using ResNet-50

2. 39 Classes trained from the PlantVillage dataset

3. Supplement Recommendations mapped to each disease

4. Multilingual UI (English, Hindi, Marathi)

5. Dynamic Rendering using Flask

6. User-friendly Interface with image upload support

## Dataset

1. Dataset Used: PlantVillage Dataset

2. Publicly available

3. ~55,000 leaf images

4. 39 manually mapped classes

5. Images include healthy & diseased leaves

6. Each class corresponds to a row in:

7. disease_info.csv

8. supplement_info.csv

## Machine Learning Approach
✔ Transfer Learning (ResNet-50)

CropCare uses ResNet-50, a deep convolutional neural network pretrained on ImageNet.

Why ResNet-50?

High accuracy

Faster convergence

Learns rich features like texture patterns

Much better performance than custom CNN

Model Changes
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 39)
Loss Function

Cross Entropy Loss:

𝐿=−∑ 𝑦 log(𝑦^)

Optimizer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)


## Image Processing Pipeline

All uploaded images undergo:

Resize → 224 × 224

Convert to Tensor

Normalization (ImageNet mean & std)

Batching for model input

Code:

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

## Model Performance
Metric	Value
Training Accuracy	97–99%
Validation Accuracy	92–94%
Loss Curve	Converges to 0.1–0.15

Trained on Google Colab using GPU.

## Technologies Used
** Backend
1. Flask
2. PyTorch
3. Torchvision
4. OpenCV
5. Pandas

** Frontend

1. HTML
2. CSS
3. JavaScript
4. Bootstrap

## Installation Guide
1️⃣ Clone the repository
git clone https://github.com/yourusername/CropCare.git
cd CropCare/App

2️⃣ Create a Virtual Environment
python -m venv venv

3️⃣ Activate Virtual Environment


Windows:

venv\Scripts\activate


Mac/Linux:

source venv/bin/activate

4️⃣ Install Dependencies
pip install -r requirements.txt

5️⃣ Run the Flask App
python app.py


Your app will start at:
http://127.0.0.1:5000/


## Contributions
1. [Aarya Patil](https://github.com/aaryapatil1729)
2. [Riya Nalamwar](https://github.com/riya-nalamwar4)
3. [Snehal Jagtap](https://github.com/snehal-subhash-jagtap)