# eco-insight

Certainly! Creating a comprehensive project like "Eco-Insight" involves multiple components, including data gathering from IoT devices, data processing, machine learning model training, and a user interface. Below is a simplified version of such a project with error handling and comments included. We'll simulate the IoT data and focus on the core aspects: data handling, model training, and prediction.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

# Simulate IoT Data
def simulate_iot_data(num_samples=1000):
    np.random.seed(42)
    # Simulating data for temperature, humidity, time of use, and electrical consumption
    temperature = np.random.normal(20, 5, num_samples) # in Celsius
    humidity = np.random.normal(50, 10, num_samples)   # in percentage
    time_of_use = np.random.choice(range(24), num_samples) # hour of the day
    consumption = (0.5 * temperature + 0.3 * humidity + 0.2 * time_of_use +
                   np.random.normal(0, 5, num_samples)) # Consumption in kWh
    data = pd.DataFrame({'temperature': temperature,
                         'humidity': humidity,
                         'time_of_use': time_of_use,
                         'consumption': consumption})
    return data

# Load and preprocess data
def load_and_preprocess_data():
    try:
        data = simulate_iot_data()
        # Shuffle the dataset
        data = data.sample(frac=1).reset_index(drop=True)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Train the machine learning model
def train_model(data):
    try:
        X = data[['temperature', 'humidity', 'time_of_use']]
        y = data['consumption']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use a Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict and evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Model Mean Squared Error: {mse}")

        # Save the model using pickle
        with open('energy_consumption_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

# Predict the energy consumption
def predict_consumption(model, temperature, humidity, time_of_use):
    try:
        X_new = np.array([[temperature, humidity, time_of_use]])
        prediction = model.predict(X_new)
        return prediction[0]
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

if __name__ == "__main__":
    # Step 1: Load and preprocess data
    data = load_and_preprocess_data()
    if data is not None:
        # Step 2: Train the model
        model = train_model(data)
        if model is not None:
            # Step 3: Predict the energy consumption for a sample input
            temperature = 22.0  # sample input
            humidity = 55.0     # sample input
            time_of_use = 18    # sample input
            prediction = predict_consumption(model, temperature, humidity, time_of_use)
            if prediction is not None:
                print(f"Predicted energy consumption: {prediction:.2f} kWh")
```

### Explanation:

1. **Simulate IoT Data**: A function generates random data to simulate household environmental metrics, given the absence of real IoT data.

2. **Data Loading and Preprocessing**: The data is shuffled to mimic real-world randomness.

3. **Model Training**: Uses a RandomForestRegressor to model energy consumption based on temperature, humidity, and time of day.

4. **Prediction**: Predicts energy consumption for given input parameters.

5. **Error Handling**: Basic error handling is embedded throughout the code, capturing exceptions and printing error messages.

Please note that this is a simplified version focused on demonstrating the basic workflow. Actual implementations would involve more sophisticated data handling, feature engineering, and IoT integrations with APIs for real-time data acquisition.