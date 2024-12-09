# Immo Eliza Regression
This is third part of the Becode 'red hill project' where a model is constructed to predict real estate prices given the data from the scraping and analysis parts.

## Installation
The project has several dependencies that can be installed through 
`pip install -r requirements.txt`.
## Usage
The `main.py` file initialized a DataLoader (takes care of all the data handling), a NeuralNetwork, and trains the network on the data returned by DataLoader. DataLoader will look for the `Clean_data.csv` file in the `data` directory.
The `main.py` can be executed by `python3 main.py`.

## Summary of the training process
The network was trained on a dataset with approximately 10 000 entries. All features were used and the addition of any external features only decreased performance. The current network architecuture
gets a test `MAE` score of `80095` and an `R2` score of `0.4993`. Bigger models like `TabTransformer` of `GANDALF` from the pytorch-tabular package (https://pytorch-tabular.readthedocs.io/en/latest/) 
did not get a better `MAE` score of `120 000`. Features that were tested but did not improve score include:
  - Cadastral income per municipality
  - Mean income per municipality
  - Population per municipality
  - Mean tax payment per municipality
  - ...
A `hyperparameterTune.py` script has been included, wich was used to tune the model hyperparams using the `keras-tuner` package.
