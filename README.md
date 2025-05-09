# 🦁 Wildlife Sustainable Classification

<div align="center">

![Wildlife Sustainable Logo](img/Screenshot%202025-04-05%20142958.png)

*Harnessing the power of AI to protect and understand our natural world*

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)

</div>

## 🌟 Overview

Wildlife Sustainable is an advanced deep learning application designed to classify wildlife species from images. Using state-of-the-art neural network architectures (GoogleNet), this project aims to assist researchers, conservationists, and wildlife enthusiasts in identifying animal species quickly and accurately.

<div align="center">

![Classification Demo](img/Screenshot%202025-04-05%20143058.png)

</div>

## 🔍 Features

- **Multi-Dataset Support**: Train and test on various animal datasets (10, 20, or 30 species)
- **Real-time Classification**: Upload and classify animal images instantly
- **Confidence Scoring**: View prediction confidence and top 5 most likely species
- **Responsive UI**: User-friendly interface that works on desktop and mobile devices
- **Extensible Architecture**: Easily add new animal classes to the model

## 🐾 Supported Animal Species

### Animal-10 Dataset
- 🐕 Cane (Dog)
- 🐎 Cavallo (Horse)
- 🐘 Elefante (Elephant)
- 🦋 Farfalla (Butterfly)
- 🐔 Gallina (Chicken)
- 🐈 Gatto (Cat)
- 🐄 Mucca (Cow)
- 🐑 Pecora (Sheep)
- 🕷️ Ragno (Spider)
- 🐿️ Scoiattolo (Squirrel)

### Additional Datasets
The application also supports expanded datasets with 20 and 30 animal species, including:

**Animal-20 additions:**
- 🦌 Antelope
- 🦇 Bat
- 🐝 Bee
- 🐗 Boar
- 🦀 Crab
- 🦊 Deer
- 🐠 Goldfish
- 🦛 Hippopotamus
- 🦑 Jellyfish
- 🦎 Lizard
- 🐁 Mouse
- 🦉 Owl
- 🐧 Penguin
- 🐖 Pig

**Animal-30 additions:**
- 🦙 Alpaca
- 🦬 American Bison
- 🦡 Badger
- 🐫 Camel
- 🦒 Giraffe
- 🦘 Kangaroo
- 🐨 Koala
- 🦝 Red Panda
- 🦏 Rhinoceros
- 🦓 Zebra

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/shkshreyas/WildLife-Sustainable
cd WildLife-Sustainable

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 📊 Model Architecture

This project utilizes the GoogleNet (Inception) architecture, a deep convolutional neural network designed for image classification tasks. The model has been trained on the Animals-10 dataset and can be extended to support additional animal classes.

```
Model: GoogleNet
Accuracy: ~93% on test set
Input Size: 224x224 pixels
Output: 10, 20, or 30 animal classes
```

## 💻 Usage

1. Start the Flask application: `python app.py`
2. Open your browser and navigate to `http://localhost:5000`
3. Upload an image of an animal using the file selector
4. Click "Classify" to process the image
5. View the prediction results and confidence scores

<div align="center">

*Example classification result*

```
Prediction: Elephant
Confidence: 98.7%

Top 5 Predictions:
1. Elephant (98.7%)
2. Horse (0.8%)
3. Cow (0.3%)
4. Dog (0.1%)
5. Cat (0.1%)
```

</div>

## 🧠 Training Your Own Model

The repository includes a Jupyter notebook (`GoogleNet_10.ipynb`) that demonstrates the training process. To train your own model:

1. Organize your dataset in the `Data` directory
2. Modify the notebook parameters as needed
3. Run the notebook to train the model
4. Export the trained model to the `model` directory

## 🔧 Technical Details

- **Backend**: Flask web server with PyTorch for inference
- **Frontend**: HTML, CSS, JavaScript with Bootstrap for styling
- **Model**: Pre-trained GoogleNet architecture fine-tuned on animal datasets
- **Image Processing**: PIL and torchvision for image transformations

## 🌱 Future Enhancements

- [ ] Add support for video classification
- [ ] Implement a mobile application
- [ ] Create an API for third-party integration
- [ ] Add geographic distribution information for identified species
- [ ] Implement conservation status indicators

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Animals-10 dataset from Kaggle
- PyTorch team for the deep learning framework
- Conservation organizations for their invaluable work protecting wildlife

---

<div align="center">

**🌍 Protecting wildlife through technology and innovation 🌍**

</div>