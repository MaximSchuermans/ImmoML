import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class DataLoader:
    '''
    Class to load and preprocess data.
    '''
    def __init__(self, file: str):
        self.filename = file

    def get_data(self) -> tuple:
        '''
        Loads the data found in parameter 'file' and returns (X,y)

        Parameters:
        Returns:
            data: tuple containing features and target
        '''
        data = pd.read_csv(self.filename)
        columns_to_drop = ['Unnamed: 0', 'id']
        data_cleaned = data.drop(columns=columns_to_drop)
        categorical_columns = ['Type of property', 'Subtype of property', 'Type of sale', 'State of the building', 'Compound Listing']
        data_encoded = pd.get_dummies(data_cleaned, columns=categorical_columns, drop_first=True)
        numerical_columns = [
            'Locality', 'Garden area', 'Surface of the land', 'Surface area of the plot of land',
            'Number of rooms', 'Living Area', 'Furnished', 'Terrace area', 'Number of facades'
        ]
        scaler = MinMaxScaler()
        data_encoded[numerical_columns] = scaler.fit_transform(data_encoded[numerical_columns])
        X = data_encoded.drop('Price', axis=1)
        y = data_encoded['Price']
        return X, y

    def get_train_test(X, y):
        '''
        Splits the data into train-validation-test sets.

        Parameters:
            X: Features
            y: target feature

        Returns:
            data: tuple of X_train, X_val, X_test, y_train, y_val, y_test
        '''
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        # Normalize numerical features
        scaler = MinMaxScaler()
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
        # Split the dataset
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test
