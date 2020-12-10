# RNA Degradation Rate Prediction
This project is intended to RNA sequence data and predict the degradation and reactivity rates under different conditions of that given sequence. 

## Dataset
### OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction
The dataset utilized for this project may be located and downloaded at the following link: https://www.kaggle.com/c/stanford-covid-vaccine/data

## Installation
Ensure the correct versions of the following libraries are downloaded:
- Python 3.6.9
- Tensorflow 2.3.0
- Keras 2.4.0
- Numpy 1.18.5, 
- Matplotlib 3.2.2 
- Pandas 1.1.4, 
- Sklearn 0.22.2

A virtual environment may also be configured using conda and the environment.yml file in our Github. 
Note: the virtual environment needs only to be installed and created once, but must be activated prior to each session of utilizing the code. 


## Usage
Once the python files are downloaded and virtual environment created and active, the code for each of the two model types may be executed as described below:
- Fully-connected Neural Network (NN) model:
     1) Open NN_main.py and ensure that the filenames and path are appropriate for where and how you have saved your dataset files. 
     2) To preprocess the data as well as train and test the model, run NN_main.py.
	```python
	python NN_main.py
	```
- RNN-based models (LSTM and GRU): 
     1) Open LSTM_GRU_main.py and ensure that the filenames and path are appropriate for where and how you have saved your dataset files. 
     2) To preprocess the data as well as train and test the model, run LSTM_GRU_main.py.
	```python
	python LSTM_GRU_main.py
	```
 

## Google Colab Implementations
Google Colab implementations of this code are available within this Github page. There is one notebook for the NN model and another for the LSTM and GRU models. 
When running with these implementations follow these steps:
- Upload desired dataset file to your own personal Google Drive.
- Update the Google Colab implementation to include the proper file path and filename.
- Ensure that GPU hardware is selected.
- Starting at the top, run each block of code in sequence. 
 
