# SAE_IA

## Information

This project has been done in the context of a school project on 3rd year of a computer science bachelor in France. The goal of this project is to create a model that can predict the behavior of a mouse using a image of the mouse. The dataset used for the training is private and can't be shared. The dataset is composed of 8 classes. The project was done to help the research in neurology. The input image are 360\*354 pixels grayscaled. The IA is hitting a 98.64% accuracy on test set with 60% training and 40% set. More information about the CNN model can be found in the model.py file.

## Installation

- Clone the repository
- Install the required library :

```bash
pip install -r requirements.txt
```

## How to use

- To run the main file you can run the main.py file while being in the src folder

```bash
cd src
python main.py
```

- If you want to run the training to get a model you can run the train.py file while being in the src folder, the dataset you are training on should be in the Dataset folder and each class should be in a separate folder inside the Dataset folder.

```bash
cd src
python train.py
```

- If you want to get the prediction result of your input image, put them in the predictions folder and run main.py, result can are printed in the console and also saved in the predictions folder.
