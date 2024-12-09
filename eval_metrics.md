# Model Report

This is a summary of the neural network that was trained for the ImmoElizaML project.

## Model

Different models and data were tried but the final model is a simple feedforward neural network with 5 hidden layer:

```python
        model = Sequential([
        Dense(64, input_dim=self.X_train.shape[1], activation='leaky_relu'),
        Dropout(0.2),
        Dense(64, activation='leaky_relu'),
        Dropout(0.2),
        Dense(128, activation='leaky_relu'),
        Dropout(0.2),
        Dense(128, activation='leaky_relu'),
        Dropout(0.2),
        Dense(64, activation='leaky_relu'),
        Dropout(0.2),
        Dense(16, activation='leaky_relu'),
        Dropout(0.2),
        Dense(1)  # Output layer
        ])
```

The model was trained with the `huber` loss and `AdamW` optimizer.

## Data

The final data is the unchanged dataset obtained from my `ImmoAnalysis` group. All features were used. The preprocessing pipeline is as follows:

- Drop `Unnamed: 0` and `id` columns.
- Check for and remove `NA` ’s.
- Encode categorical columns with the pandas `get_dummies` function.
- Split the data into train, validation, and test sets. Each split is `70-30`.
- Fit a normalizer on the training data and normalize train-val-test sets.

A complete list of the features used:

```python
categorical_columns = ['Type of property', 'Subtype of property', 'Type of sale'\
, 'State of the building', 'Compound Listing']

numerical_columns = [
'Locality', 'Garden area', 'Surface of the land'\
, 'Surface area of the plot of land',
'Number of rooms', 'Living Area', 'Furnished', 'Terrace area'\
, 'Number of facades'
]
```

## Score metrics

- Train `MAE`: `77287` ; Test `MAE`: `78635`
- Train `RMSE`: `125796` ; Test `RMSE` : `115590`
- Train `R2` : `0.5000` ; Test `R2` : `0.4992`
- Train `MAPE`: `23.5336` ; Test `MAPE`: `23.6717`

### Thing that did not work

- Adding external features like `mean income, mean tax, population, living condition score` per municipality.
- More complex models like `GANDAlF`  and `TabTransformer`.
- Transfer learning on other datasets.
