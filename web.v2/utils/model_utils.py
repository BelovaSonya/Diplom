import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import pickle

class ParkingPredictor:
    def __init__(self, cnn_model, rf_lstm_model):
        self.cnn_model = cnn_model
        self.rf_lstm_model = rf_lstm_model
        
    def preprocess_features(self, parking_id, time_of_day, day_of_week, travel_time):
        """Подготовка признаков для моделей"""
        features = {
            'parking_id': hash(parking_id) % 1000,
            'hour': time_of_day,
            'day_of_week': day_of_week,
            'travel_time': travel_time,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'is_rush_hour': 1 if (8 <= time_of_day <= 10 or 17 <= time_of_day <= 19) else 0
        }
        
        # Нормализация
        features['hour_norm'] = features['hour'] / 24
        features['travel_time_norm'] = min(1, features['travel_time'] / 120)
        
        return features
    
    def predict_cnn(self, features):
        """Прогноз с помощью CNN"""
        if self.cnn_model is None:
            return 0.7  # Возвращаем значение по умолчанию
        
        # Создаем входные данные для CNN
        input_data = np.array([[
            features['hour_norm'],
            features['travel_time_norm'],
            features['is_rush_hour'],
            features['is_weekend']
        ]])
        
        try:
            prediction = self.cnn_model.predict(input_data, verbose=0)
            return float(prediction[0][0])
        except:
            return 0.7
    
    def predict_rf_lstm(self, features, historical_data=None):
        """Прогноз с помощью RF+LSTM"""
        if self.rf_lstm_model is None:
            return 0.65
        
        try:
            # Для LSTM нужна последовательность данных
            # Если есть исторические данные, используем их
            if historical_data is not None and len(historical_data) >= 10:
                sequence = np.array(historical_data[-10:])
                lstm_input = sequence.reshape(1, 10, 1)
                # Здесь должен быть вызов LSTM модели
                # lstm_pred = self.rf_lstm_model['lstm'].predict(lstm_input)
                # return float(lstm_pred[0][0])
            
            # Иначе используем RF часть
            rf_features = np.array([[
                features['hour_norm'],
                features['travel_time_norm'],
                features['is_rush_hour'],
                features['is_weekend']
            ]])
            
            # return float(self.rf_lstm_model['rf'].predict(rf_features)[0])
            return 0.65
        except:
            return 0.65
    
    def ensemble_predict(self, parking_id, travel_time):
        """Ансамблевое предсказание"""
        now = datetime.now()
        time_of_day = now.hour + now.minute / 60
        day_of_week = now.weekday()
        
        features = self.preprocess_features(parking_id, time_of_day, day_of_week, travel_time)
        
        # Получаем предсказания от обеих моделей
        cnn_pred = self.predict_cnn(features)
        rflstm_pred = self.predict_rf_lstm(features)
        
        # Ансамбль с весами из метаданных
        # Веса из вашего файла: rf_weight=0.7, lstm_weight=0.3
        final_prediction = 0.7 * cnn_pred + 0.3 * rflstm_pred
        
        return final_prediction