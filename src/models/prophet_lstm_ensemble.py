import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import tensorflow as tf
from keras.api.models import Sequential, load_model
from keras.api.layers import LSTM, Dense, Dropout
from keras.api.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import joblib
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)

class ProphetLSTMEnsemble:
    """
    Mô hình kết hợp Prophet và LSTM cho dự đoán chuỗi thời gian COVID-19
    với khả năng lưu và tải mô hình
    """

    def __init__(self, ensemble_method='weighted_average',
                 prophet_weight=0.5, lstm_weight=0.5, stacking=False):
        """
        Khởi tạo mô hình kết hợp
        
        Tham số:
        ensemble_method (str): Phương pháp kết hợp ('weighted_average' hoặc 'stacking')
        prophet_weight (float): Trọng số cho dự đoán của Prophet (trường hợp weighted_average)
        lstm_weight (float): Trọng số cho dự đoán của LSTM (trường hợp weighted_average)
        stacking (bool): Có sử dụng dự đoán của Prophet làm đặc trưng cho LSTM hay không
        """
        self.ensemble_method = ensemble_method
        self.prophet_weight = prophet_weight
        self.lstm_weight = lstm_weight
        self.stacking = stacking
        
        self.prophet_model = None
        self.lstm_model = None
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.look_back = 7
        
        os.makedirs('models', exist_ok=True)

    def load_and_prepare_data(self, file_path, country, variable='Confirmed'):
        """
        Đọc và chuẩn bị dữ liệu cho một quốc gia cụ thể
        """
        df = pd.read_csv(file_path)
        
        country_df = df[df['Country/Region'] == country]
        
        if len(country_df) == 0:
            raise ValueError(f"Không tìm thấy dữ liệu cho quốc gia {country}")
        
        row = country_df.iloc[0]
        
        id_vars = ['Country/Region', 'Lat', 'Long']
        date_columns = [col for col in df.columns if col not in id_vars]
        
        data = []
        for date_col in date_columns:
            data.append({
                'Date': pd.to_datetime(date_col, format='%m/%d/%y'),
                variable: row[date_col]
            })
        
        time_series_df = pd.DataFrame(data)
        
        return time_series_df
    
    def split_train_test(self, data, test_size=30):
        """
        Chia dữ liệu thành tập train và test
        """
        last_date = data['Date'].max()
        
        train_data = data[data['Date'] < (last_date - pd.Timedelta(days=test_size))].copy()
        test_data = data[data['Date'] >= (last_date - pd.Timedelta(days=test_size))].copy()
        
        return train_data, test_data
    
    def prepare_prophet_data(self, data, target_col):
        """
        Chuẩn bị dữ liệu cho model Prophet
        """
        prophet_df = data[['Date', target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        return prophet_df

    def prepare_lstm_data(self, data, look_back=None):
        """
        Chuẩn bị dữ liệu cho model LSTM
        """
        if look_back is None:
            look_back = self.look_back
            
        values = data.values
        scaled_values = self.scaler.fit_transform(values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(len(scaled_values) - look_back):
            X.append(scaled_values[i:(i + look_back), 0])
            y.append(scaled_values[i + look_back, 0])
        
        X, y = np.array(X), np.array(y)
        
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y, scaled_values
    

    def prepare_stacking_data(self, prophet_predictions, original_data, look_back=None):
        """
        Chuẩn bị dữ liệu cho mô hình stacking
        """
        if look_back is None:
            look_back = self.look_back
        
        combined_data = np.column_stack([
            original_data,
            prophet_predictions
        ])
        
        scaled_values = self.scaler.fit_transform(combined_data)
        
        X, y = [], []
        for i in range(len(scaled_values) - look_back):
            X.append(scaled_values[i:(i + look_back), :])
            y.append(scaled_values[i + look_back, 0])
        
        X, y = np.array(X), np.array(y)
        
        return X, y, scaled_values
    
    def build_lstm_model(self, input_shape, units=50):
        """
        Xây dựng mô hình LSTM
        """
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(units=units))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def fit_prophet(self, train_data, test_periods=30, future_periods=60):
        """
        Huấn luyện mô hình Prophet
        """
        print("Đang huấn luyện mô hình Prophet...")
        
        self.prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        
        self.prophet_model.add_seasonality(name='weekly', period=7, fourier_order=3)
        
        self.prophet_model.fit(train_data)
        
        future = self.prophet_model.make_future_dataframe(periods=test_periods + future_periods)
        self.prophet_forecast = self.prophet_model.predict(future)
        
        return self.prophet_forecast
    
    def fit_lstm(self, X_train, y_train, validation_split=0.2, epochs=100, batch_size=32):
        """
        Huấn luyện mô hình LSTM
        """
        print("Đang huấn luyện mô hình LSTM...")
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        self.lstm_model = self.build_lstm_model(input_shape)
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.lstm_model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )
        
        return history
    
    def fit(self, train_data, target_col, test_periods=30, future_periods=60, 
            lstm_epochs=100, lstm_batch_size=32):
        """
        Huấn luyện cả hai mô hình Prophet và LSTM
        """
        prophet_data = self.prepare_prophet_data(train_data, target_col)
        
        prophet_forecast = self.fit_prophet(prophet_data, test_periods, future_periods)
        
        prophet_train_pred = prophet_forecast.iloc[:len(train_data)]['yhat'].values
        
        if self.stacking:
            X_train, y_train, _ = self.prepare_stacking_data(
                prophet_train_pred, 
                train_data[target_col].values, 
                self.look_back
            )
        else:
            X_train, y_train, _ = self.prepare_lstm_data(
                train_data[target_col].values,
                self.look_back
            )
        
        lstm_history = self.fit_lstm(
            X_train, y_train, 
            epochs=lstm_epochs, 
            batch_size=lstm_batch_size
        )
        
        return {
            'prophet': self.prophet_model,
            'lstm': self.lstm_model,
            'prophet_forecast': prophet_forecast,
            'lstm_history': lstm_history
        }

    def save_models(self, country, target_col):
        """
        Lưu các mô hình đã huấn luyện vào thư mục models
        """
        if self.prophet_model is None or self.lstm_model is None:
            raise ValueError("Các mô hình chưa được huấn luyện")
        
        base_filename = f"{country}_{target_col}"
        
        model_dir = os.path.join('models', base_filename)
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, 'prophet_model.pkl'), 'wb') as f:
            pickle.dump(self.prophet_model, f)
        
        self.lstm_model.save(os.path.join(model_dir, 'lstm_model.keras'))
        
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        
        params = {
            'ensemble_method': self.ensemble_method,
            'prophet_weight': self.prophet_weight,
            'lstm_weight': self.lstm_weight,
            'stacking': self.stacking,
            'look_back': self.look_back,
            'data_info': {
                'country': country,
                'target_col': target_col
            }
        }
        
        with open(os.path.join(model_dir, 'params.json'), 'w') as f:
            json.dump(params, f)
        
        if hasattr(self, 'prophet_forecast'):
            self.prophet_forecast.to_csv(os.path.join(model_dir, 'prophet_forecast.csv'), index=False)
        
        print(f"Đã lưu mô hình thành công vào thư mục: {model_dir}")
        
        return model_dir

    def load_models(self, country, target_col):
        """
        Tải lại mô hình từ thư mục models
        """
        base_filename = f"{country}_{target_col}"
        model_dir = os.path.join('models', base_filename)
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Không tìm thấy mô hình cho {country} và {target_col}")
        
        with open(os.path.join(model_dir, 'prophet_model.pkl'), 'rb') as f:
            self.prophet_model = pickle.load(f)
        
        lstm_path = os.path.join(model_dir, 'lstm_model.keras')
        
        if not os.path.exists(lstm_path):
            lstm_path = os.path.join(model_dir, 'lstm_model.h5')
        
        if not os.path.exists(lstm_path):
            lstm_path = os.path.join(model_dir, 'lstm_model')
        
        try:
            self.lstm_model = load_model(lstm_path)
        except Exception as e:
            print(f"Error loading LSTM model: {e}")
            try:
                from keras.api.layers import TFSMLayer
                self.lstm_model = TFSMLayer(lstm_path, call_endpoint='serving_default')
            except Exception as load_error:
                print(f"Failed to load LSTM model: {load_error}")
                raise
        
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
        with open(os.path.join(model_dir, 'params.json'), 'r') as f:
            params = json.load(f)
        
        self.ensemble_method = params['ensemble_method']
        self.prophet_weight = params['prophet_weight']
        self.lstm_weight = params['lstm_weight']
        self.stacking = params['stacking']
        self.look_back = params['look_back']
        
        prophet_forecast_path = os.path.join(model_dir, 'prophet_forecast.csv')
        if os.path.exists(prophet_forecast_path):
            self.prophet_forecast = pd.read_csv(prophet_forecast_path)
            self.prophet_forecast['ds'] = pd.to_datetime(self.prophet_forecast['ds'])
        
        print(f"Đã tải mô hình thành công từ thư mục: {model_dir}")
        
        return {
            'prophet': self.prophet_model,
            'lstm': self.lstm_model,
            'params': params
        }
    
    def check_model_exists(self, country, target_col):
        """
        Kiểm tra xem mô hình đã tồn tại hay chưa
        """
        base_filename = f"{country}_{target_col}"
        model_dir = os.path.join('models', base_filename)
        
        return os.path.exists(model_dir)
    
    def predict_prophet(self, periods=30):
        """
        Dự đoán với mô hình Prophet
        """
        if self.prophet_model is None:
            raise ValueError("Mô hình Prophet chưa được huấn luyện")
            
        return self.prophet_forecast.iloc[-periods:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def predict_lstm(self, input_data, periods=30):
        """
        Dự đoán với mô hình LSTM
        """
        if self.lstm_model is None:
            raise ValueError("Mô hình LSTM chưa được huấn luyện")
        
        predictions = []
        current_input = input_data
        
        for _ in range(periods):
            next_pred = self.lstm_model.predict(current_input)
            predictions.append(next_pred[0, 0])
            
            current_input = np.roll(current_input, -1, axis=1)
            current_input[0, -1, 0] = next_pred
        
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        return predictions.flatten()
    
    def predict(self, test_data, target_col, periods=30):
        """
        Dự đoán kết hợp từ cả hai mô hình
        """
        # Predict using Prophet
        prophet_predictions = self.predict_prophet(periods)
        prophet_values = prophet_predictions['yhat'].values
        
        # Prepare input for LSTM
        if self.stacking:
            # Get last sequence of actual values
            last_sequence = test_data[target_col].values[-self.look_back:]
            
            # Get corresponding prophet predictions for the same period
            prophet_last_sequence = prophet_values[:self.look_back]
            
            # Combine actual and prophet predictions
            combined_input = np.column_stack([last_sequence, prophet_last_sequence])
            
            # Transform combined input
            scaled_input = self.scaler.transform(combined_input)
            
            # Reshape for LSTM (batch_size, timesteps, features)
            lstm_input = scaled_input.reshape(1, self.look_back, 2)
            
            # LSTM prediction with multi-feature input
            lstm_predictions = []
            current_input = lstm_input.copy()
            
            for _ in range(periods):
                # Predict next value
                next_pred = self.lstm_model.predict(current_input)
                lstm_predictions.append(next_pred[0, 0])
                
                # Update input sequence
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1, 0] = next_pred
                current_input[0, -1, 1] = prophet_values[len(lstm_predictions)-1]
            
            # Inverse transform predictions
            lstm_predictions = self.scaler.inverse_transform(
                np.column_stack([
                    np.array(lstm_predictions), 
                    prophet_values[:len(lstm_predictions)]
                ])
            )[:, 0]
        
        else:
            # Non-stacking scenario
            last_sequence = test_data[target_col].values[-self.look_back:]
            
            # Transform last sequence
            scaled_input = self.scaler.transform(last_sequence.reshape(-1, 1))
            
            # Reshape for LSTM
            lstm_input = scaled_input.reshape(1, self.look_back, 1)
            
            # LSTM prediction
            lstm_predictions = []
            current_input = lstm_input.copy()
            
            for _ in range(periods):
                # Predict next value
                next_pred = self.lstm_model.predict(current_input)
                lstm_predictions.append(next_pred[0, 0])
                
                # Update input sequence
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1, 0] = next_pred
            
            # Inverse transform predictions
            lstm_predictions = self.scaler.inverse_transform(
                np.array(lstm_predictions).reshape(-1, 1)
            ).flatten()
        
        # Ensemble predictions
        if self.ensemble_method == 'weighted_average':
            combined_predictions = (
                self.prophet_weight * prophet_values +
                self.lstm_weight * lstm_predictions
            )
        else:
            combined_predictions = (prophet_values + lstm_predictions) / 2
        
        # Create future dates
        future_dates = pd.date_range(
            start=test_data['Date'].max() + pd.Timedelta(days=1), 
            periods=periods
        )
        
        # Construct result DataFrame
        result_df = pd.DataFrame({
            'Date': future_dates,
            'Prophet_Prediction': prophet_values,
            'LSTM_Prediction': lstm_predictions,
            'Ensemble_Prediction': combined_predictions,
            'Lower_CI': prophet_predictions['yhat_lower'].values,
            'Upper_CI': prophet_predictions['yhat_upper'].values
        })
        
        return result_df
    
    def evaluate(self, test_data, target_col, prophet_predictions=None, lstm_predictions=None, ensemble_predictions=None):
        """
        Đánh giá hiệu suất các mô hình
        """
        test_values = test_data[target_col].values

        def calculate_metrics(true_values, predictions, model_name):
            if predictions is None or len(predictions) == 0:
                return None
            
            min_length = min(len(true_values), len(predictions))
            true_values = true_values[:min_length]
            predictions = predictions[:min_length]
            
            mae = mean_absolute_error(true_values, predictions)
            rmse = np.sqrt(mean_squared_error(true_values, predictions))
            
            mape = np.mean(np.abs((true_values - predictions) / (true_values + 1e-10))) * 100
            
            return {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }

        metrics = {}
        
        if prophet_predictions is not None:
            metrics['Prophet'] = calculate_metrics(test_values, prophet_predictions, 'Prophet')
        
        if lstm_predictions is not None:
            metrics['LSTM'] = calculate_metrics(test_values, lstm_predictions, 'LSTM')
        
        if ensemble_predictions is not None:
            metrics['Ensemble'] = calculate_metrics(test_values, ensemble_predictions, 'Ensemble')
        
        return metrics
    
    def plot_results(self, historical_data, test_data, forecast_data, target_col):
        """
        Vẽ biểu đồ kết quả dự đoán
        """
        plt.figure(figsize=(14, 8))
        
        plt.plot(historical_data['Date'], historical_data[target_col], label='Dữ liệu lịch sử', color='blue')
        
        plt.plot(test_data['Date'], test_data[target_col], label='Dữ liệu test', color='green')
        
        plt.plot(forecast_data['Date'], forecast_data['Prophet_Prediction'], 
                label='Dự đoán Prophet', color='red', linestyle='--')
        
        plt.plot(forecast_data['Date'], forecast_data['LSTM_Prediction'], 
                label='Dự đoán LSTM', color='orange', linestyle='--')
        
        plt.plot(forecast_data['Date'], forecast_data['Ensemble_Prediction'], 
                label='Dự đoán Ensemble', color='purple', linewidth=2)
        
        plt.fill_between(
            forecast_data['Date'],
            forecast_data['Lower_CI'],
            forecast_data['Upper_CI'],
            color='red', alpha=0.2, label='Khoảng tin cậy 95%'
        )
        
        plt.title(f'Dự đoán {target_col} với mô hình kết hợp Prophet-LSTM')
        plt.xlabel('Ngày')
        plt.ylabel(target_col)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        return plt

    def run_forecast(self, file_path, country, variable='Confirmed', 
                     test_size=30, future_periods=60, lstm_epochs=100,
                     force_retrain=False, save_model=True):
        """
        Chạy toàn bộ qui trình dự đoán cho một quốc gia
        
        Tham số mới:
        force_retrain (bool): Có bắt buộc huấn luyện lại mô hình không
        save_model (bool): Có lưu mô hình sau khi huấn luyện không
        """
        print(f"===== Dự đoán COVID-19 cho {country} sử dụng mô hình kết hợp Prophet-LSTM =====")
        
        # Kiểm tra xem mô hình đã tồn tại chưa
        model_exists = self.check_model_exists(country, variable)
        
        if model_exists and not force_retrain:
            print(f"Tìm thấy mô hình đã huấn luyện cho {country} và {variable}. Đang tải mô hình...")
            self.load_models(country, variable)
        else:
            if model_exists and force_retrain:
                print(f"Bắt buộc huấn luyện lại mô hình cho {country} và {variable}...")
            else:
                print(f"Không tìm thấy mô hình đã huấn luyện. Đang huấn luyện mô hình mới...")
            
            data = self.load_and_prepare_data(file_path, country, variable)
            
            train_data, test_data = self.split_train_test(data, test_size)
            
            print(f"Dữ liệu train: {len(train_data)} ngày từ {train_data['Date'].min()} đến {train_data['Date'].max()}")
            print(f"Dữ liệu test: {len(test_data)} ngày từ {test_data['Date'].min()} đến {test_data['Date'].max()}")
            
            self.fit(train_data, variable, test_size, future_periods, lstm_epochs)
            
            if save_model:
                self.save_models(country, variable)
        
        data = self.load_and_prepare_data(file_path, country, variable)
        train_data, test_data = self.split_train_test(data, test_size)
        
        prophet_test_predictions = self.prophet_forecast.iloc[-test_size-future_periods:-future_periods]['yhat'].values
        
        if self.stacking:
            prophet_train_pred = self.prophet_forecast.iloc[:len(train_data)]['yhat'].values
            X_train, y_train, scaled_train = self.prepare_stacking_data(
                prophet_train_pred, 
                train_data[variable].values, 
                self.look_back
            )
            
            lstm_test_predictions = []
            
            last_sequence = np.column_stack([
                train_data[variable].values[-self.look_back:],
                prophet_train_pred[-self.look_back:]
            ])
            current_input = self.scaler.transform(last_sequence).reshape(1, self.look_back, 2)
            
            for _ in range(test_size):
                next_pred = self.lstm_model.predict(current_input)
                lstm_test_predictions.append(next_pred[0, 0])
                
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1, 0] = next_pred
                current_input[0, -1, 1] = next_pred
            
            lstm_test_predictions = self.scaler.inverse_transform(
                np.array(lstm_test_predictions).reshape(-1, 1)
            ).flatten()
        else:
            X_train, y_train, scaled_train = self.prepare_lstm_data(
                train_data[variable].values,
                self.look_back
            )
            
            lstm_test_predictions = []
            
            last_sequence = scaled_train[-self.look_back:].reshape(1, self.look_back, 1)
            
            for _ in range(test_size):
                next_pred = self.lstm_model.predict(last_sequence)
                lstm_test_predictions.append(next_pred[0, 0])
                
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = next_pred
            
            lstm_test_predictions = self.scaler.inverse_transform(
                np.array(lstm_test_predictions).reshape(-1, 1)
            ).flatten()
        
        ensemble_test_predictions = (
            self.prophet_weight * prophet_test_predictions +
            self.lstm_weight * lstm_test_predictions
        )
        
        metrics = self.evaluate(
            test_data, 
            variable, 
            prophet_test_predictions, 
            lstm_test_predictions, 
            ensemble_test_predictions
        )
        
        print("\nĐánh giá mô hình trên tập test:")
        for model_name, model_metrics in metrics.items():
            print(f"{model_name}:")
            for metric_name, metric_value in model_metrics.items():
                print(f"  {metric_name}: {metric_value:.2f}")
        
        future_forecast = self.predict(data, variable, future_periods)
        
        self.plot_results(train_data, test_data, future_forecast, variable)
        plt.show()
        
        return {
            'prophet_model': self.prophet_model,
            'lstm_model': self.lstm_model,
            'test_metrics': metrics,
            'future_forecast': future_forecast
        }
