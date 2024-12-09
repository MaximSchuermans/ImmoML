from data.get_data import DataLoader
from model import NeuralNetwork

dataloader = DataLoader('./data/Clean_data.csv')
model = NeuralNetwork(*dataloader.get_train_test(dataloader.get_data()))
