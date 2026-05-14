# ensemble_rf_lstm_fixed.py
"""
Комбинированная модель прогнозирования: Random Forest + LSTM (ИСПРАВЛЕННАЯ ВЕРСИЯ)
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import config


class EnsembleRFLSTM:
    """
    Комбинированная модель прогнозирования:
    - Random Forest для краткосрочного прогноза (следующие 1-2 интервала)
    - LSTM для долгосрочного прогноза (следующие 6-12 интервалов)
    - Ансамбль: взвешенное усреднение предсказаний
    """
    
    def __init__(self, lookback=12, forecast_horizon=6, model_dir=None):
        """
        Args:
            lookback: количество предыдущих временных шагов для прогноза
            forecast_horizon: горизонт прогнозирования (количество шагов вперед)
            model_dir: директория для сохранения моделей
        """
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.model_dir = model_dir or config.MODELS_DIR
        
        # Инициализация моделей
        self.rf_model = None
        self.lstm_model = None
        
        # Скейлеры
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
        # Веса для ансамбля (можно настроить)
        self.rf_weight = 0.4   # Random Forest вес
        self.lstm_weight = 0.6  # LSTM вес
        
        self.is_trained = False
        
        # Создаем директорию для сохранения
        os.makedirs(self.model_dir, exist_ok=True)
        
    def create_sequences(self, data):
        """
        Создание последовательностей для обучения LSTM
        
        Args:
            data: массив временного ряда (уже нормализованный)
        
        Returns:
            X: входные последовательности (samples, lookback, 1)
            y: целевые значения (samples, forecast_horizon)
        """
        X, y = [], []
        for i in range(len(data) - self.lookback - self.forecast_horizon + 1):
            X.append(data[i:i + self.lookback])
            y.append(data[i + self.lookback:i + self.lookback + self.forecast_horizon])
        
        X = np.array(X)
        y = np.array(y)
        
        # Изменяем форму X для LSTM: (samples, lookback, 1)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return X, y
    
    def create_features_for_rf(self, time_series, timestamps=None):
        """
        Создание признаков для Random Forest из временного ряда
        
        Args:
            time_series: временной ряд (значения загрузки)
            timestamps: временные метки
        
        Returns:
            X_features: матрица признаков
            y_target: целевые значения
        """
        features = []
        targets = []
        
        for i in range(len(time_series) - self.forecast_horizon):
            # Используем lookback предыдущих значений
            start_idx = max(0, i - self.lookback + 1)
            window = time_series[start_idx:i + 1]
            
            # Если окно меньше lookback, дополняем первым значением
            if len(window) < self.lookback:
                pad_width = self.lookback - len(window)
                window = np.pad(window, (pad_width, 0), 'edge')
            
            # Статистические признаки
            feature_row = list(window)
            
            # Дополнительные признаки
            feature_row.append(np.mean(window))           # среднее
            feature_row.append(np.std(window))            # стандартное отклонение
            feature_row.append(np.max(window))            # максимум
            feature_row.append(np.min(window))            # минимум
            feature_row.append(window[-1] - window[0])    # разница между последним и первым
            feature_row.append(np.median(window))         # медиана
            
            # Тренд (линейная регрессия)
            if len(window) > 2:
                x_idx = np.arange(len(window))
                slope = np.polyfit(x_idx, window, 1)[0]
                feature_row.append(slope)
            else:
                feature_row.append(0)
            
            # Добавляем временные признаки (если есть временные метки)
            if timestamps is not None and i < len(timestamps):
                dt = timestamps[i]
                if isinstance(dt, (datetime, pd.Timestamp)):
                    feature_row.append(dt.hour)           # час
                    feature_row.append(dt.weekday())      # день недели
                    feature_row.append(1 if dt.weekday() >= 5 else 0)  # выходной
                else:
                    feature_row.extend([0, 0, 0])
            else:
                feature_row.extend([0, 0, 0])
            
            features.append(feature_row)
            targets.append(time_series[i + self.forecast_horizon])
        
        return np.array(features), np.array(targets)
    
    def train_rf(self, time_series, timestamps=None):
        """
        Обучение Random Forest модели
        
        Args:
            time_series: временной ряд
            timestamps: временные метки
        """
        print("\n" + "="*50)
        print("Обучение Random Forest модели")
        print("="*50)
        
        X, y = self.create_features_for_rf(time_series, timestamps)
        
        if len(X) == 0:
            print("Ошибка: недостаточно данных для обучения RF")
            return False
        
        # Разделение на train/test
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Масштабирование
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # Обучение
        self.rf_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=0
        )
        
        self.rf_model.fit(X_train_scaled, y_train)
        
        # Оценка
        y_pred = self.rf_model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nRandom Forest результаты:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  R2: {r2:.4f}")
        print(f"  Train size: {len(X_train)}")
        print(f"  Test size: {len(X_test)}")
        
        # Сохраняем модель
        rf_path = os.path.join(self.model_dir, "RF_Ensemble.pkl")
        joblib.dump(self.rf_model, rf_path)
        joblib.dump(self.scaler_X, os.path.join(self.model_dir, "RF_scaler.pkl"))
        
        return True
    
    def build_lstm_model(self, input_shape):
        """
        Построение архитектуры LSTM
        
        Args:
            input_shape: форма входных данных (lookback, 1)
        """
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(self.forecast_horizon)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_lstm(self, time_series):
        """
        Обучение LSTM модели
        
        Args:
            time_series: временной ряд (исходные значения, не нормализованные)
        """
        print("\n" + "="*50)
        print("Обучение LSTM модели")
        print("="*50)
        
        # Нормализуем данные для LSTM
        time_series_reshaped = time_series.reshape(-1, 1)
        time_series_normalized = self.scaler_y.fit_transform(time_series_reshaped).flatten()
        
        # Создаем последовательности
        X, y = self.create_sequences(time_series_normalized)
        
        if len(X) == 0:
            print("Ошибка: недостаточно данных для обучения LSTM")
            return False
        
        # Разделение на train/test
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Построение модели
        self.lstm_model = self.build_lstm_model((self.lookback, 1))
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                os.path.join(self.model_dir, "LSTM_Ensemble.keras"),
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # Обучение
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Оценка (восстанавливаем исходный масштаб)
        y_pred_scaled = self.lstm_model.predict(X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        y_test_original = self.scaler_y.inverse_transform(y_test)
        
        mae = mean_absolute_error(y_test_original, y_pred)
        mse = mean_squared_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original.flatten(), y_pred.flatten())
        
        print(f"\nLSTM результаты:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  R2: {r2:.4f}")
        print(f"  Train size: {len(X_train)}")
        print(f"  Test size: {len(X_test)}")
        
        return True
    
    def train(self, time_series, timestamps=None):
        """
        Обучение обеих моделей
        
        Args:
            time_series: временной ряд (значения загрузки)
            timestamps: временные метки
        """
        # Преобразуем в numpy массив если нужно
        if isinstance(time_series, pd.Series):
            time_series = time_series.values
        elif isinstance(time_series, list):
            time_series = np.array(time_series)
        
        # Обучаем Random Forest
        rf_success = self.train_rf(time_series, timestamps)
        
        # Обучаем LSTM
        lstm_success = self.train_lstm(time_series)
        
        self.is_trained = rf_success and lstm_success
        
        # Сохраняем веса и параметры
        self._save_metadata()
        
        return self.is_trained
    
    def predict_rf(self, recent_values, timestamps=None):
        """
        Прогноз с помощью Random Forest
        
        Args:
            recent_values: последние lookback значений
            timestamps: временные метки для текущего момента
        
        Returns:
            прогноз на следующий шаг
        """
        if self.rf_model is None:
            return None
        
        # Подготовка признаков
        window = list(recent_values[-self.lookback:])
        
        if len(window) < self.lookback:
            window = [window[0]] * (self.lookback - len(window)) + window
        
        feature_row = list(window)
        feature_row.append(np.mean(window))
        feature_row.append(np.std(window))
        feature_row.append(np.max(window))
        feature_row.append(np.min(window))
        feature_row.append(window[-1] - window[0])
        feature_row.append(np.median(window))
        
        # Тренд
        if len(window) > 2:
            x_idx = np.arange(len(window))
            slope = np.polyfit(x_idx, window, 1)[0]
            feature_row.append(slope)
        else:
            feature_row.append(0)
        
        # Временные признаки
        if timestamps is not None:
            dt = timestamps
            if isinstance(dt, (datetime, pd.Timestamp)):
                feature_row.append(dt.hour)
                feature_row.append(dt.weekday())
                feature_row.append(1 if dt.weekday() >= 5 else 0)
            else:
                feature_row.extend([0, 0, 0])
        else:
            feature_row.extend([0, 0, 0])
        
        # Масштабирование и предсказание
        features_scaled = self.scaler_X.transform([feature_row])
        prediction = self.rf_model.predict(features_scaled)[0]
        
        return prediction
    
    def predict_lstm(self, recent_values):
        """
        Прогноз с помощью LSTM
        
        Args:
            recent_values: последние lookback значений (в исходном масштабе)
        
        Returns:
            прогноз на forecast_horizon шагов вперед (в исходном масштабе)
        """
        if self.lstm_model is None:
            return None
        
        # Подготовка входных данных
        window = list(recent_values[-self.lookback:])
        
        if len(window) < self.lookback:
            window = [window[0]] * (self.lookback - len(window)) + window
        
        # Нормализуем window
        window_array = np.array(window).reshape(-1, 1)
        window_scaled = self.scaler_y.transform(window_array).flatten()
        
        # Изменяем форму для LSTM: (1, lookback, 1)
        X_input = window_scaled.reshape(1, self.lookback, 1)
        
        # Предсказание (в нормализованном масштабе)
        pred_scaled = self.lstm_model.predict(X_input, verbose=0)[0]
        
        # Возвращаем в исходный масштаб
        prediction = self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        
        return prediction
    
    def predict(self, recent_values, timestamps=None, return_all=False):
        """
        Комбинированный прогноз (ансамбль)
        
        Args:
            recent_values: последние значения временного ряда
            timestamps: временные метки
            return_all: вернуть все прогнозы (RF, LSTM, ансамбль)
        
        Returns:
            ансамблевый прогноз
        """
        # Прогноз RF (только следующий шаг)
        rf_pred = self.predict_rf(recent_values, timestamps)
        
        # Прогноз LSTM (весь горизонт)
        lstm_preds = self.predict_lstm(recent_values)
        
        if rf_pred is None and lstm_preds is None:
            return None
        
        # Комбинируем прогнозы
        ensemble_preds = []
        
        for i in range(self.forecast_horizon):
            if rf_pred is not None and lstm_preds is not None and i == 0:
                # Для первого шага используем взвешенное среднее
                ensemble_val = self.rf_weight * rf_pred + self.lstm_weight * lstm_preds[i]
            elif lstm_preds is not None:
                # Для остальных шагов используем LSTM
                ensemble_val = lstm_preds[i]
            else:
                ensemble_val = rf_pred
            
            # Ограничиваем значения в диапазоне 0-100
            ensemble_val = max(0, min(100, ensemble_val))
            ensemble_preds.append(ensemble_val)
        
        if return_all:
            return {
                'random_forest': rf_pred,
                'lstm': lstm_preds,
                'ensemble': np.array(ensemble_preds)
            }
        
        return np.array(ensemble_preds)
    
    def _save_metadata(self):
        """Сохранение метаданных модели"""
        metadata = {
            'lookback': self.lookback,
            'forecast_horizon': self.forecast_horizon,
            'rf_weight': self.rf_weight,
            'lstm_weight': self.lstm_weight,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(self.model_dir, "RF_LSTM_metadata.pkl")
        joblib.dump(metadata, metadata_path)
    
    def load_models(self):
        """Загрузка сохраненных моделей"""
        rf_path = os.path.join(self.model_dir, "RF_Ensemble.pkl")
        lstm_path = os.path.join(self.model_dir, "LSTM_Ensemble.keras")
        scaler_path = os.path.join(self.model_dir, "RF_scaler.pkl")
        
        if os.path.exists(rf_path):
            self.rf_model = joblib.load(rf_path)
            print("✓ Random Forest загружен")
        
        if os.path.exists(lstm_path):
            self.lstm_model = load_model(lstm_path)
            print("✓ LSTM загружена")
        
        if os.path.exists(scaler_path):
            self.scaler_X = joblib.load(scaler_path)
            print("✓ Скейлер RF загружен")
        
        # Загружаем метаданные
        metadata_path = os.path.join(self.model_dir, "RF_LSTM_metadata.pkl")
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.lookback = metadata.get('lookback', self.lookback)
            self.forecast_horizon = metadata.get('forecast_horizon', self.forecast_horizon)
            self.rf_weight = metadata.get('rf_weight', self.rf_weight)
            self.lstm_weight = metadata.get('lstm_weight', self.lstm_weight)
            self.is_trained = metadata.get('is_trained', False)
            print("✓ Метаданные загружены")
        
        return self.rf_model is not None and self.lstm_model is not None


class TimeSeriesDataLoader:
    """Загрузчик данных для временных рядов"""
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.data = None
        
    def load_from_csv(self, csv_path):
        """Загрузка из CSV файла"""
        try:
            self.data = pd.read_csv(csv_path)
            
            # Определяем колонки с временем и значениями
            time_col = None
            value_col = None
            
            for col in self.data.columns:
                if 'time' in col.lower() or 'timestamp' in col.lower():
                    time_col = col
                if 'occupancy' in col.lower() or 'rate' in col.lower():
                    value_col = col
            
            if time_col and value_col:
                self.data['timestamp'] = pd.to_datetime(self.data[time_col])
                self.data = self.data.sort_values('timestamp')
                return self.data['timestamp'].values, self.data[value_col].values
            
            return None, None
        except Exception as e:
            print(f"Ошибка загрузки CSV: {e}")
            return None, None
    
    def generate_synthetic_data(self, n_points=500, start_date=None):
        """
        Генерация синтетических данных для тестирования
        
        Args:
            n_points: количество точек
            start_date: начальная дата
        """
        if start_date is None:
            start_date = datetime(2024, 1, 1, 8, 0, 0)
        
        timestamps = [start_date + timedelta(minutes=10 * i) for i in range(n_points)]
        
        # Генерация реалистичного временного ряда
        np.random.seed(42)
        
        # Базовый уровень (дневная сезонность)
        hours = [ts.hour for ts in timestamps]
        base_occupancy = 30 + 40 * np.sin(np.array(hours) * np.pi / 12 - np.pi)
        
        # Добавляем шум
        noise = np.random.normal(0, 5, n_points)
        
        # Добавляем автокорреляцию
        for i in range(1, n_points):
            base_occupancy[i] = 0.7 * base_occupancy[i] + 0.3 * base_occupancy[i-1]
        
        # Нормируем в диапазон 0-100
        occupancy = np.clip(base_occupancy + noise, 0, 100)
        
        return timestamps, occupancy


class EnsemblePredictorVisualizer:
    """Визуализатор для комбинированной модели"""
    
    def __init__(self, ensemble_model):
        self.ensemble = ensemble_model
        
    def plot_predictions(self, actual_values, predictions, title="Прогноз загрузки парковки"):
        """Визуализация прогнозов"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Прогноз на горизонт
        ax1 = axes[0, 0]
        ax1.plot(range(len(actual_values)), actual_values, 'b-', label='Фактические', linewidth=2)
        
        # Добавляем прогноз
        pred_start = len(actual_values) - 1
        pred_x = list(range(pred_start, pred_start + len(predictions) + 1))
        pred_y = [actual_values[-1]] + list(predictions)
        ax1.plot(pred_x, pred_y, 'r--', label='Прогноз', linewidth=2, marker='o')
        
        ax1.set_title(f'{title} (Горизонт: {len(predictions)} шагов)', fontsize=12)
        ax1.set_xlabel('Временной шаг')
        ax1.set_ylabel('Загрузка (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Сравнение моделей
        ax2 = axes[0, 1]
        ax2.text(0.5, 0.7, f'Random Forest\nВес: {self.ensemble.rf_weight}', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='lightblue'))
        ax2.text(0.5, 0.3, f'LSTM\nВес: {self.ensemble.lstm_weight}', 
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightgreen'))
        ax2.set_title('Состав ансамбля', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        # 3. Важность признаков Random Forest
        ax3 = axes[1, 0]
        if hasattr(self.ensemble, 'rf_model') and self.ensemble.rf_model:
            importances = self.ensemble.rf_model.feature_importances_
            n_features = min(15, len(importances))
            indices = np.argsort(importances)[-n_features:]
            
            feature_names = [f'F{i}' for i in indices]
            ax3.barh(range(n_features), importances[indices])
            ax3.set_yticks(range(n_features))
            ax3.set_yticklabels(feature_names)
            ax3.set_xlabel('Важность')
            ax3.set_title('Важность признаков (Random Forest)', fontsize=12)
        else:
            ax3.text(0.5, 0.5, 'RF не обучена', ha='center', va='center')
        
        # 4. Параметры модели
        ax4 = axes[1, 1]
        ax4.text(0.5, 0.5, f'Параметры модели\n'
                           f'Lookback: {self.ensemble.lookback}\n'
                           f'Forecast: {self.ensemble.forecast_horizon}\n'
                           f'RF вес: {self.ensemble.rf_weight}\n'
                           f'LSTM вес: {self.ensemble.lstm_weight}',
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightyellow'))
        ax4.set_title('Конфигурация', fontsize=12)
        
        plt.tight_layout()
        
        # Сохранение
        save_path = os.path.join(config.RESULTS_DIR, 'ensemble_rf_lstm_predictions.png')
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nГрафик сохранен: {save_path}")
        
        plt.show()
        
    def plot_forecast_comparison(self, rf_pred, lstm_pred, ensemble_pred, actual_next=None):
        """Сравнение прогнозов разных моделей"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        steps = range(len(ensemble_pred))
        
        if lstm_pred is not None:
            ax.plot(steps, lstm_pred, 'g--', label='LSTM прогноз', linewidth=2, marker='s', markersize=4)
        
        if rf_pred is not None:
            ax.axhline(y=rf_pred, color='orange', linestyle='--', label=f'RF прогноз: {rf_pred:.1f}%', linewidth=2)
        
        ax.plot(steps, ensemble_pred, 'r-', label='Ансамбль (RF+LSTM)', linewidth=2, marker='o', markersize=6)
        
        if actual_next is not None:
            ax.scatter([0], [actual_next], color='blue', s=100, zorder=5, 
                      label=f'Фактическое: {actual_next:.1f}%')
        
        ax.set_xlabel('Шаг прогноза', fontsize=12)
        ax.set_ylabel('Загрузка парковки (%)', fontsize=12)
        ax.set_title('Сравнение прогнозов: Random Forest vs LSTM vs Ансамбль', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Добавляем информацию о весах
        ax.text(0.02, 0.98, f'RF вес: {self.ensemble.rf_weight} | LSTM вес: {self.ensemble.lstm_weight}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        save_path = os.path.join(config.RESULTS_DIR, 'ensemble_forecast_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранен: {save_path}")
        
        plt.show()


def main():
    """Главная функция для тестирования комбинированной модели"""
    
    print("\n" + "="*70)
    print("КОМБИНИРОВАННАЯ МОДЕЛЬ ПРОГНОЗИРОВАНИЯ")
    print("Random Forest + LSTM")
    print("="*70)
    
    # Параметры модели
    LOOKBACK = 12      # Используем 12 предыдущих шагов (120 минут при шаге 10 мин)
    FORECAST_HORIZON = 6  # Прогнозируем 6 шагов вперед (60 минут)
    
    print(f"\nПараметры модели:")
    print(f"  Lookback (окно истории): {LOOKBACK} шагов")
    print(f"  Горизонт прогноза: {FORECAST_HORIZON} шагов")
    
    # Создаем загрузчик данных
    loader = TimeSeriesDataLoader()
    
    # Пытаемся загрузить реальные данные
    data_path = os.path.join(config.RESULTS_DIR, "parking_occupancy_full.csv")
    timestamps = None
    occupancy = None
    
    if os.path.exists(data_path):
        print(f"\nЗагрузка данных из: {data_path}")
        timestamps, occupancy = loader.load_from_csv(data_path)
        if occupancy is not None:
            print(f"  Загружено {len(occupancy)} записей")
            print(f"  Диапазон загрузки: {occupancy.min():.1f}% - {occupancy.max():.1f}%")
            print(f"  Средняя загрузка: {occupancy.mean():.1f}%")
    
    # Если нет реальных данных, генерируем синтетические
    if occupancy is None or len(occupancy) < 100:
        print("\nРеальные данные не найдены. Генерируем синтетические данные...")
        timestamps, occupancy = loader.generate_synthetic_data(n_points=500)
        print(f"  Сгенерировано {len(occupancy)} записей")
        print(f"  Диапазон загрузки: {occupancy.min():.1f}% - {occupancy.max():.1f}%")
    
    # Создаем комбинированную модель
    ensemble = EnsembleRFLSTM(
        lookback=LOOKBACK,
        forecast_horizon=FORECAST_HORIZON
    )
    
    # Обучение
    print("\n" + "="*70)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("="*70)
    
    success = ensemble.train(occupancy, timestamps)
    
    if success:
        print("\n✅ Модель успешно обучена!")
        
        # Сохраняем модели
        print(f"\nМодели сохранены в: {ensemble.model_dir}")
        
        # Тестирование на последних данных
        print("\n" + "="*70)
        print("ТЕСТИРОВАНИЕ МОДЕЛИ")
        print("="*70)
        
        # Берем последние LOOKBACK значений для прогноза
        recent_values = occupancy[-LOOKBACK:]
        current_time = timestamps[-1] if timestamps is not None else None
        
        # Делаем прогноз
        predictions = ensemble.predict(recent_values, current_time, return_all=True)
        
        if predictions:
            print(f"\nПрогноз на следующие {FORECAST_HORIZON} шагов (каждый шаг ~10 минут):")
            print("-" * 60)
            print(f"{'Шаг':<6} {'RF':<12} {'LSTM':<12} {'Ансамбль':<12}")
            print("-" * 60)
            
            for i in range(FORECAST_HORIZON):
                rf_val = predictions['random_forest'] if i == 0 else "-"
                lstm_val = f"{predictions['lstm'][i]:.1f}%" if predictions['lstm'] is not None else "-"
                ensemble_val = f"{predictions['ensemble'][i]:.1f}%"
                
                rf_str = f"{rf_val:.1f}%" if isinstance(rf_val, float) else rf_val
                print(f"{i+1:<6} {rf_str:<12} {lstm_val:<12} {ensemble_val:<12}")
            
            print("-" * 60)
            
            # Визуализация
            visualizer = EnsemblePredictorVisualizer(ensemble)
            visualizer.plot_predictions(occupancy[-30:], predictions['ensemble'])
            visualizer.plot_forecast_comparison(
                predictions['random_forest'],
                predictions['lstm'],
                predictions['ensemble']
            )
            
            # Вывод рекомендаций
            print("\n" + "="*70)
            print("РЕКОМЕНДАЦИИ НА ОСНОВЕ ПРОГНОЗА")
            print("="*70)
            
            current_occupancy = occupancy[-1]
            next_occupancy = predictions['ensemble'][0]
            
            print(f"\nТекущая загрузка: {current_occupancy:.1f}%")
            print(f"Прогноз через ~10 минут: {next_occupancy:.1f}%")
            
            if next_occupancy < current_occupancy:
                print("\n📉 Прогнозируется снижение загрузки. Рекомендуется:")
                print("   • Можно подождать 10-15 минут для поиска лучшего места")
            elif next_occupancy > current_occupancy:
                print("\n📈 Прогнозируется увеличение загрузки. Рекомендуется:")
                print("   • Рекомендуется припарковаться сейчас")
                print("   • Через 20-30 минут мест может не быть")
            else:
                print("\n➡ Загрузка стабильна. Рекомендуется:")
                print("   • Текущая ситуация сохранится")
            
            if predictions['ensemble'][-1] > 80:
                print("\n⚠ ВНИМАНИЕ: Через час ожидается высокая загрузка!")
            elif predictions['ensemble'][-1] < 30:
                print("\n✅ Через час ожидается низкая загрузка. Отличное время для парковки!")
    else:
        print("\n❌ Ошибка при обучении модели")
    
    print("\n" + "="*70)
    print("ЗАВЕРШЕНИЕ РАБОТЫ")
    print("="*70)


if __name__ == "__main__":
    main()