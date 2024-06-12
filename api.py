from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense
from pymongo import MongoClient
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)

def download_and_save_csv():
    client = MongoClient('mongodb+srv://orlandosykes:orlandosykes@technorcluster.0ayow.mongodb.net/PengmasPasca?retryWrites=true&w=majority')
    db = client['PengmasPasca']
    collection = db['monitors']

    cursor = collection.find()
    data = pd.DataFrame(list(cursor))

    if 'createdAt' in data.columns:
        data['createdAt'] = pd.to_datetime(data['createdAt'], errors='coerce')
    else:
        raise KeyError("'createdAt' column is missing from the data")

    data = data.dropna(subset=['createdAt'])

    csv_file_path = 'data.csv'
    data.to_csv(csv_file_path, index=False)
    return csv_file_path

def preprocess_data(csv_file_path):
    data = pd.read_csv(csv_file_path)
    
    features = ['salinity', 'acidity', 'oxygen', 'temperature']
    for feature in features:
        data[feature] = data[feature].replace(0, np.nan)
        data[feature] = data[feature].astype(float)

    data = data.dropna()

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[features])

    return data, data_scaled, scaler

def create_dataset(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), :])
        y.append(data[i + time_steps, :]) 
    return np.array(X), np.array(y)

def calculate_performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    val_accuracy = 100 - mape
    return mse, mae, val_accuracy

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model = load_model('lstm_model.h5')
    except:
        return jsonify({"message": "Model not found"}), 404
    
    try:
        csv_file_path = download_and_save_csv()
        data_interpolated, data_scaled, scaler = preprocess_data(csv_file_path)
        time_steps = 1
        X, y = create_dataset(data_scaled, time_steps)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        y_pred_test = model.predict(X_test).tolist()
        y_pred_train = model.predict(X_train).tolist()

        mse_test, mae_test, val_accuracy_test = calculate_performance(y_test, np.array(y_pred_test))
        mse_train, mae_train, val_accuracy_train = calculate_performance(y_train, np.array(y_pred_train))

        results = {
            'mse_test': mse_test,
            'mae_test': mae_test,
            'val_accuracy_test': val_accuracy_test,
            'mse_train': mse_train,
            'mae_train': mae_train,
            'val_accuracy_train': val_accuracy_train,
            'y_pred_test': y_pred_test,
            'y_pred_train': y_pred_train
        }
        
        return jsonify(results), 200
    
    except Exception as e:
        return jsonify({"message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
