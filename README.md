Install Python
Install pip installer if necessary
pip install h5py
pip install keras
Unzip the zip files under Training/DataPrep/DatasetPictures

cd Training/DataPrep
python ProcessImg.py

Install Tensorflow: https://www.tensorflow.org/get_started/os_setup
Then run SimpleEquationCNN.py under Training

To test your own images:
Upload photos to QuickStart/Uploads
run:
python LoadNetwork.py (name of picture file here)
