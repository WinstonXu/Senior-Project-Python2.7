Install Python 2.7
https://www.python.org/downloads/

Go to command line and type these commands:
pip install h5py
pip install keras
pip install Pillow

Unzip the zip files under Training/DataPrep/DatasetPictures
-Should have two folders: Mnist and Operators

In command line, go back to the base directory(Senior-Project-Python2.7)
type cd Training/DataPrep
type python ProcessImg.py

Install Tensorflow: https://www.tensorflow.org/get_started/os_setup
type python SimpleEquationCNN.py
or if you want to use Theano: either type
KERAS_BACKEND=theano python SimpleEquationCNN.py
or use these instructions https://keras.io/backend/

To test your own images:
Upload photos to QuickStart/Uploads
run:
from base directory: cd Quickstart
type python LoadNetwork.py (name of picture file here)
