'''
________________________________________________________________________________________

This is a script that does a random search in parameter space using the keras-tune package.
The model is stored in the directory 'best_price_model' as an hdf5 file.
________________________________________________________________________________________


'''
def build_model(hp):
    # Define a Sequential model
    model = Sequential()
    
    # First hidden layer with variable number of units
    model.add(Dense(
        units=hp.Int('units_1', min_value=32, max_value=128, step=32),
        input_dim=X_train.shape[1],  # Adjust based on the shape of your dataset
        activation=hp.Choice('activation_1', values=['relu', 'leaky_relu', 'tanh'])  # Hyperparameterize activation function
    ))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))  # Hyperparameterize dropout rate

    # Additional hidden layers with variable sizes and dropout rates
    for i in range(2, 8):
        model.add(Dense(
            units=hp.Int(f'units_{i}', min_value=16, max_value=128, step=16),
            activation=hp.Choice(f'activation_{i}', values=['relu', 'leaky_relu', 'tanh'])  # Hyperparameterize activation function
        ))
        model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))  # Hyperparameterize dropout rate

    # Output layer for regression (single unit)
    model.add(Dense(1))

    # Compile the model with an Adam optimizer and Mean Squared Error loss
    model.compile(optimizer=Adam(), loss='huber', metrics=['mae'])

    return model

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,  # Number of models to try
    executions_per_trial=1,  # Run each model configuration 2 times
    directory='my_dir4',
    project_name='price_prediction'
)

tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=32)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best number of units in the first layer: {best_hps.get('units_1')}")
print(f"Best learning rate: {best_hps.get('learning_rate')}")

# Build the best model
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)
test_loss, test_mae, test_r2 = best_model.evaluate(X_test, y_test) # Include test_r2 to capture the third return value (r2_score)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}, Test R2: {test_r2}") # Print the R2 score along with the other metrics.
best_model.save('best_price_model.h5')
