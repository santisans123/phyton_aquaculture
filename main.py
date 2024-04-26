import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import gdown
import io
import requests

def metrics():
    # Unduh file CSV dari Google Drive
    file_url = 'https://drive.google.com/uc?id=1xtolcRrwcvOSgdylrv-UoMJaFXMvKor0'
    gdown.download(file_url, 'data.csv', quiet=False)

    # Baca data dari file CSV
    response = requests.get(file_url)
    data = pd.read_csv(io.StringIO(response.content.decode('utf-8')), delimiter=',')
    print(data.columns)
    print(data.head())

    # Cleaning data yang bernilai 0
    data = data.replace(0, np.nan)

    # Interpolasi untuk mengisi nilai-nilai yang hilang
    data_interpolated = data.interpolate(method='linear', axis=0)

    # Cek apakah masih ada NaN setelah interpolasi
    if data_interpolated.isnull().values.any():
        print("Warning: Still contains NaN after interpolation. Handling NaN values.")
        # Hapus baris yang mengandung NaN
        data_interpolated = data_interpolated.dropna()

    # Parsing data waktu dan konversi ke format numerik
    data_interpolated['Waktu'] = pd.to_datetime(data_interpolated['Waktu'])
    data_interpolated['Waktu_numeric'] = data_interpolated['Waktu'].astype(np.int64)

    # Pilih kolom yang akan digunakan sebagai fitur
    features = ['Waktu_numeric', 'Suhu', 'Oksigen', 'Salinitas', 'pH']

    # Hapus titik (.) dan konversi ke float
    for feature in features[1:]:
        data_interpolated[feature] = data_interpolated[feature].replace('\.', '', regex=True).astype(float)

    # Normalisasi data menggunakan Min-Max scaling
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_interpolated[features])

    # Tentukan time_steps sebagai 1
    time_steps = 1

    def create_dataset(data, time_steps=1):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps), :])
            y.append(data[i + time_steps:i + time_steps + 1, 1:])  # Hanya satu baris (sebanyak fitur)
        return np.array(X), np.array(y)

    X, y = create_dataset(data_scaled, time_steps)

    # Split data menjadi data latih dan data uji dengan proporsi 80%:20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Buat dan latih model LSTM
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units=4))  # Jumlah fitur output sesuai dengan jumlah fitur (Suhu, Oksigen, Salinitas, pH)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Perbarui penggunaan seluruh data aktual untuk training
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Prediksi dengan model yang telah dilatih
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Invers transform hasil prediksi untuk mendapatkan nilai asli
    y_pred_test_original = scaler.inverse_transform(np.concatenate((X_test[:, -1, 0].reshape(-1, 1), y_pred_test.reshape(-1, 4)), axis=1))[:, 1:]
    y_pred_train_original = scaler.inverse_transform(np.concatenate((X_train[:, -1, 0].reshape(-1, 1), y_pred_train.reshape(-1, 4)), axis=1))[:, 1:]

    # Buat tanggal untuk data testing
    waktu_batas = data_interpolated['Waktu'].iloc[-1]
    tanggal_testing = pd.date_range(start=waktu_batas, periods=len(y_test) + 1, freq='D')[1:]

    # Evaluasi model dengan menghitung MSE, MAE, dan MAPE
    mse_lstm = mean_squared_error(y_test.flatten(), y_pred_test.flatten())
    mae_lstm = mean_absolute_error(y_test.flatten(), y_pred_test.flatten())

    # Perhitungan MAPE dengan menangani pembagian dengan nol
    mape_lstm = np.mean(np.abs((y_test.flatten() - y_pred_test.flatten()) / np.maximum(y_test.flatten(), 1))) * 100

    # Menampilkan hasil evaluasi
    print(f'Mean Squared Error (MSE) on test data: {mse_lstm}')
    print(f'Mean Absolute Error (MAE) on test data: {mae_lstm}')
    print(f'Mean Absolute Percentage Error (MAPE) on test data: {mape_lstm:.2f}%')

    metrics = {
        "mse": mse_lstm,
        "mae": mae_lstm,
        "mape": mape_lstm
    }
    return metrics