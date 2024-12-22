Traffic Sign Classification

This project is a Traffic Sign Recognition system that uses a Convolutional Neural Network (CNN) model to classify images of traffic signs into 43 categories. The application also features a user-friendly GUI to classify uploaded images.

Features
Train a CNN Model: Trains a deep learning model using labeled traffic sign images.
Real-Time Image Classification: Classifies uploaded images using the trained model.
Graphical User Interface (GUI): A Tkinter-based GUI for easy interaction.
Visualization: Accuracy and loss graphs for training and validation.
Requirements
Install the necessary libraries before running the project:

pip install numpy pandas matplotlib scikit-learn tensorflow keras pillow opencv-python

Project Structure
Model Training and Saving:

Load traffic sign data from the dataset.
Preprocess images by resizing them to 30x30 pixels.
Split the data into training and testing sets (80%-20%).
Train a CNN model with 15 epochs and save it as my_model.h5.
Test Dataset Evaluation:

Load and preprocess test images specified in a Test.csv file.
Evaluate the saved model on the test set and calculate its accuracy.
GUI for Real-Time Image Classification:

A Tkinter GUI allows users to upload and classify traffic sign images using the trained model.
Displays the classification result and the associated traffic sign label.
How the Code Works
Step 1: Data Preparation
The training images are stored in folders named train/0, train/1, ..., train/42, where each folder represents a specific traffic sign class.
Images are resized to 30x30 and converted into numpy arrays for training.
The labels are one-hot encoded for multi-class classification.
Step 2: Model Training
A CNN model is built using the following layers:
Convolutional Layers: Extract features from images.
MaxPooling Layers: Reduce spatial dimensions and computational complexity.
Dropout Layers: Prevent overfitting by randomly disabling neurons.
Dense Layer: Fully connected layers for classification.
Softmax Layer: Outputs probabilities for 43 traffic sign classes.
The model is trained using categorical_crossentropy loss and the Adam optimizer.
After training, the model is saved as my_model.h5.
Step 3: Testing the Model
The test images, specified in Test.csv, are loaded, resized, and converted into numpy arrays.
The trained model is loaded and used to predict the class of each test image.
The accuracy of the model on the test dataset is calculated.
Step 4: GUI Implementation
A Tkinter-based GUI allows users to:
Upload an image of a traffic sign.
Classify the image by clicking the "Classify Image" button.
The GUI displays the predicted traffic sign label using a dictionary mapping (classes).


Training the Model
Place training data in a train directory with subfolders for each class (0-42).
Run the script to train and save the model:
bash
Ensure my_model.h5 is in the same directory as the GUI script.

Upload an image and classify it.

Dataset

This project uses the German Traffic Sign Recognition Benchmark (GTSRB). Ensure you download and structure the dataset before running the code.

https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?resource=download

1) Install archive and unzip the files for training part.
2) Run Train.py code. 
3) Use same archive files for ınterface part, but create another folder ın the name of " Traffic sign classification" and upload files in it.
4) Run İnterface.py code. 
