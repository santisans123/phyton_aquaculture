from flask import Flask, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pymongo import MongoClient
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

app = Flask(__name__)

def download_and_save_csv():
    client = MongoClient('mongodb+srv://orlandosykes:orlandosykes@technorcluster.0ayow.mongodb.net/PengmasPasca?retryWrites=true&w=majority')
    db = client['PengmasPasca']
    collection = db['monitors']

    cursor = collection.find()
    data = pd.DataFrame(list(cursor))

    # Ensure 'Waktu' is a datetime column
    if 'createdAt' in data.columns:
        data['createdAt'] = pd.to_datetime(data['createdAt'], errors='coerce')
    else:
        raise KeyError("'createdAt' column is missing from the data")

    # Drop rows where 'Waktu' could not be parsed
    data = data.dropna(subset=['createdAt'])

    # Save data to a CSV file
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
        y.append(data[i + time_steps, :])  # Mengambil seluruh kolom fitur sebagai target
    return np.array(X), np.array(y)

def calculate_performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
    val_accuracy = 100 - mape
    return mse, mae, val_accuracy

def plot_predictions_with_dates(y_true, y_pred, features, dates):
    fig = go.Figure()
    for i, feature in enumerate(features):
        fig.add_trace(go.Scatter(x=dates, y=y_true[:, i], mode='lines+markers', name=f'Actual {feature}'))
        fig.add_trace(go.Scatter(x=dates, y=y_pred[:, i], mode='lines+markers', name=f'Predicted {feature}'))

    fig.update_layout(
        title='LSTM Time Series Prediction',
        xaxis_title='Time',
        yaxis_title='Values',
        width=1200,  # Set the width of the plot to 1200 pixels
    )
    graph_html = fig.to_html(full_html=False)
    return graph_html

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['GET'])
def train():
    csv_file_path = download_and_save_csv()
    data_interpolated, data_scaled, scaler = preprocess_data(csv_file_path)
    time_steps = 1
    X, y = create_dataset(data_scaled, time_steps)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units=4))  # Mengubah units menjadi 4
    model.compile(optimizer='adam', loss='mean_squared_error')

    print(model.summary())  # Cetak ringkasan model untuk verifikasi

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    model.save('lstm_model.h5')
    
    return render_template('train.html', message='Model trained and saved successfully')

@app.route('/predict', methods=['GET'])
def predict():
    from tensorflow.keras.models import load_model
    model = load_model('lstm_model.h5')
    
    csv_file_path = download_and_save_csv()
    data_interpolated, data_scaled, scaler = preprocess_data(csv_file_path)
    time_steps = 1
    X, y = create_dataset(data_scaled, time_steps)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    mse_test, mae_test, val_accuracy_test = calculate_performance(y_test, y_pred_test)
    mse_train, mae_train, val_accuracy_train = calculate_performance(y_train, y_pred_train)

    results = {
        'mse_test': mse_test,
        'mae_test': mae_test,
        'val_accuracy_test': val_accuracy_test,
        'mse_train': mse_train,
        'mae_train': mae_train,
        'val_accuracy_train': val_accuracy_train
    }

    features = ['salinity', 'acidity', 'oxygen', 'temperature']

    # Generate dates for the testing data
    waktu_batas = pd.to_datetime(data_interpolated['createdAt'].iloc[-1])
    tanggal_testing = pd.date_range(start=waktu_batas, periods=len(y_test) + 1, freq='D')[1:]

    graph_html = plot_predictions_with_dates(y_test, y_pred_test, features, tanggal_testing)

    return render_template('predict.html', results=results, graph_html=graph_html)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
