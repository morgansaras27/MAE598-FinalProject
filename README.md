# RNA Degradation Rate Prediction
This project is intended to RNA sequence data and predict the degradation and reactivity rates under different conditions of that given sequence. 

## Dataset
### OpenVaccine: COVID-19 mRNA Vaccine Degradation Prediction
The dataset utilized for this project may be located and downloaded at the following link: https://www.kaggle.com/c/stanford-covid-vaccine/data

## Installation
The virtual environment must be installed in order to ensure the proper libraries, pacakages, and versions are utilized to run the model. To do this, follow the below instructions:
1. Install conda
```bash
pip install conda
```
2. Create a new virtual environment using the environment.yml file located in our GitHub
```bash
conda env create -f environment.yml
```
3. Activate the virual environment (named MAE598_FinalProject)
```bash
conda activate MAE598
```
Note: the virtual environment needs only to be installed and created once, but must be activated prior to each session of utilizing the code. 


## Usage
Once the python files are downloaded and virtual environment created and active, the code for each of the two model types may be executed as described below:
- Fully-connected Neural Network (NN) model:
     1) 1) Open NN_main.py and ensure that the filenames and path are appropriate for where and how you have saved your dataset files. 
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
Google Colab implementations of this code are available within this Github page.
When running with these implementations follow these steps:
     1) Upload desired dataset file to your own personal Google Drive.
     2) Update the Google Colab implementation to include the proper file path and filename.
     3) Starting at the top, run each block of code in sequence. 
 
