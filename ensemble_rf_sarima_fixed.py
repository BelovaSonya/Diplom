# ensemble_rf_sarima_fixed.py
"""
Комбинированная модель прогнозирования: Random Forest + SARIMA (ИСПРАВЛЕННАЯ)
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

import joblib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import config


class EnsembleRFSARIMA:
    """
    Комбинированная модель прогнозирования: Random Forest + SARIMA
    """
    
    def __init__(self, 
                 lookback=24, 
                 forecast_horizon=12,
                 seasonal_period=24,
                 model_dir=None,
                 ensemble_type='weighted'):
        
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.seasonal_period = seasonal_period
        self.model_dir = model_dir or config.MODELS_DIR
        self.ensemble_type = ensemble_type
        
        self.sarima_model = None
        self.sarima_results = None
        self.rf_model = None
        self.rf_residual_model = None
        
        self.scaler_X = StandardScaler()
        
        self.sarima_weight = 0.5
        self.rf_weight = 0.5
        
        self.sarima_order = (1, 1, 1)
        self.sarima_seasonal_order = (1, 1, 1, seasonal_period)
        
        self.is_trained = False
        
        os.makedirs(self.model_dir, exist_ok=True)
    
    def check_stationarity(self, series, title='Временной ряд'):
        """Проверка стационарности ряда"""
        result = adfuller(series.dropna())
        print(f'\nТест Дики-Фуллера для {title}:')
        print(f'  ADF Statistic: {result[0]:.4f}')
        print(f'  p-value: {result[1]:.4f}')
        print(f'  Ряд {"стационарен" if result[1] < 0.05 else "не стационарен"}')
        return result[1] < 0.05
    
    def train_sarima(self, series):
        """Обучение SARIMA модели"""
        print("\n" + "="*50)
        print("Обучение SARIMA модели")
        print("="*50)
        
        self.check_stationarity(series, 'Исходный ряд')
        
        print(f"\nОбучение SARIMA с параметрами:")
        print(f"  order: {self.sarima_order}")
        print(f"  seasonal_order: {self.sarima_seasonal_order}")
        
        try:
            self.sarima_model = SARIMAX(
                series,
                order=self.sarima_order,
                seasonal_order=self.sarima_seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.sarima_results = self.sarima_model.fit(disp=False)
            print(f"\n✓ SARIMA обучена успешно")
            print(f"  AIC: {self.sarima_results.aic:.2f}")
            print(f"  BIC: {self.sarima_results.bic:.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при обучении SARIMA: {e}")
            return False
    
    def create_features_for_rf(self, time_series, residuals=None, timestamps=None):
        """
        Создание признаков для Random Forest
        
        Args:
            time_series: временной ряд (numpy array)
            residuals: остатки SARIMA
            timestamps: временные метки
        """
        # Преобразуем в numpy array если это Series
        if isinstance(time_series, pd.Series):
            time_series = time_series.values
        if isinstance(residuals, pd.Series):
            residuals = residuals.values
        
        features = []
        targets = []
        
        for i in range(self.lookback, len(time_series) - self.forecast_horizon):
            # Лаговые значения (используем список, а не Series)
            window = time_series[i - self.lookback:i].tolist()
            
            # Статистические признаки
            feature_row = window.copy()
            feature_row.append(float(np.mean(window)))
            feature_row.append(float(np.std(window)))
            feature_row.append(float(np.max(window)))
            feature_row.append(float(np.min(window)))
            feature_row.append(float(window[-1] - window[0]))  # изменение
            feature_row.append(float(np.median(window)))
            
            # Скользящие средние
            ma6 = float(np.mean(window[-6:])) if len(window) >= 6 else float(np.mean(window))
            ma12 = float(np.mean(window[-12:])) if len(window) >= 12 else float(np.mean(window))
            feature_row.append(ma6)
            feature_row.append(ma12)
            
            # Тренд
            if len(window) > 2:
                x_idx = np.arange(len(window))
                slope = float(np.polyfit(x_idx, window, 1)[0])
            else:
                slope = 0.0
            feature_row.append(slope)
            
            # Волатильность
            if len(window) > 1:
                volatility = float(np.std(np.diff(window)))
            else:
                volatility = 0.0
            feature_row.append(volatility)
            
            # Остатки SARIMA
            if residuals is not None and len(residuals) > i:
                resid_window = residuals[i - self.lookback:i]
                if len(resid_window) > 0:
                    feature_row.append(float(np.mean(resid_window)))
                    feature_row.append(float(np.std(resid_window)))
                    feature_row.append(float(resid_window[-1]))
                else:
                    feature_row.extend([0.0, 0.0, 0.0])
            else:
                feature_row.extend([0.0, 0.0, 0.0])
            
            # Временные признаки
            if timestamps is not None and i < len(timestamps):
                dt = timestamps[i]
                if isinstance(dt, (datetime, pd.Timestamp)):
                    feature_row.append(float(dt.hour))
                    feature_row.append(float(dt.weekday()))
                    feature_row.append(1.0 if dt.weekday() >= 5 else 0.0)
                    feature_row.append(float(dt.month))
                else:
                    feature_row.extend([0.0, 0.0, 0.0, 0.0])
            else:
                feature_row.extend([0.0, 0.0, 0.0, 0.0])
            
            features.append(feature_row)
            targets.append(float(time_series[i + self.forecast_horizon]))
        
        return np.array(features), np.array(targets)
    
    def train_random_forest(self, time_series, residuals=None, timestamps=None):
        """Обучение Random Forest модели"""
        print("\n" + "="*50)
        print("Обучение Random Forest модели")
        print("="*50)
        
        # Преобразуем в numpy array
        if isinstance(time_series, pd.Series):
            time_series = time_series.values
        
        X, y = self.create_features_for_rf(time_series, residuals, timestamps)
        
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
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
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
        print(f"  Количество признаков: {X.shape[1]}")
        print(f"  Train size: {len(X_train)}")
        print(f"  Test size: {len(X_test)}")
        
        return True
    
    def train_residual_model(self, time_series, timestamps=None):
        """Обучение RF для прогнозирования остатков SARIMA"""
        print("\n" + "="*50)
        print("Обучение модели остатков (Cascade подход)")
        print("="*50)
        
        if isinstance(time_series, pd.Series):
            time_series = time_series.values
        
        # Получаем предсказания SARIMA
        try:
            sarima_pred = self.sarima_results.predict(start=0, end=len(time_series)-1)
            if isinstance(sarima_pred, pd.Series):
                sarima_pred = sarima_pred.values
            residuals = time_series - sarima_pred
        except Exception as e:
            print(f"Ошибка получения остатков: {e}")
            residuals = np.zeros(len(time_series))
        
        X, _ = self.create_features_for_rf(time_series, residuals, timestamps)
        
        if len(X) == 0:
            print("Ошибка: недостаточно данных")
            return False
        
        # Цель - предсказать остаток
        y = residuals[self.lookback:len(time_series) - self.forecast_horizon]
        y = y[:len(X)]
        
        # Разделение
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Масштабирование
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # Обучение
        self.rf_residual_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_residual_model.fit(X_train_scaled, y_train)
        
        # Оценка
        y_pred = self.rf_residual_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\nМодель остатков:")
        print(f"  MAE остатков: {mae:.4f}")
        
        return True
    
    def train(self, time_series, timestamps=None, optimize_sarima=False):
        """Обучение комбинированной модели"""
        # Преобразуем в pandas Series для SARIMA
        if isinstance(time_series, np.ndarray):
            time_series_series = pd.Series(time_series)
        else:
            time_series_series = pd.Series(time_series)
        
        # 1. Обучение SARIMA
        sarima_success = self.train_sarima(time_series_series)
        
        # 2. Обучение Random Forest
        if self.ensemble_type == 'cascade' and sarima_success:
            rf_success = self.train_residual_model(time_series, timestamps)
        else:
            rf_success = self.train_random_forest(time_series, None, timestamps)
        
        self.is_trained = (sarima_success or rf_success)
        
        # Сохраняем метаданные
        self._save_metadata()
        
        return self.is_trained
    
    def predict_sarima(self, steps=None):
        """Прогноз с помощью SARIMA"""
        if self.sarima_results is None:
            return None
        
        steps = steps or self.forecast_horizon
        
        try:
            forecast = self.sarima_results.forecast(steps=steps)
            if isinstance(forecast, pd.Series):
                forecast = forecast.values
            return forecast
        except Exception as e:
            print(f"Ошибка прогноза SARIMA: {e}")
            return None
    
    def predict_random_forest(self, recent_values, timestamps=None, residuals=None):
        """Прогноз с помощью Random Forest"""
        if self.rf_model is None:
            return None
        
        # Преобразуем в список
        if isinstance(recent_values, (np.ndarray, pd.Series)):
            recent_values = recent_values.tolist()
        
        # Подготовка признаков
        window = recent_values[-self.lookback:] if len(recent_values) >= self.lookback else recent_values
        
        if len(window) < self.lookback:
            window = [window[0]] * (self.lookback - len(window)) + window
        
        feature_row = window.copy()
        feature_row.append(float(np.mean(window)))
        feature_row.append(float(np.std(window)))
        feature_row.append(float(np.max(window)))
        feature_row.append(float(np.min(window)))
        feature_row.append(float(window[-1] - window[0]))
        feature_row.append(float(np.median(window)))
        
        # Скользящие средние
        ma6 = float(np.mean(window[-6:])) if len(window) >= 6 else float(np.mean(window))
        ma12 = float(np.mean(window[-12:])) if len(window) >= 12 else float(np.mean(window))
        feature_row.append(ma6)
        feature_row.append(ma12)
        
        # Тренд
        if len(window) > 2:
            x_idx = np.arange(len(window))
            slope = float(np.polyfit(x_idx, window, 1)[0])
        else:
            slope = 0.0
        feature_row.append(slope)
        
        # Волатильность
        if len(window) > 1:
            volatility = float(np.std(np.diff(window)))
        else:
            volatility = 0.0
        feature_row.append(volatility)
        
        # Остатки
        if residuals is not None and len(residuals) > 0:
            feature_row.append(float(np.mean(residuals[-self.lookback:])) if len(residuals) >= self.lookback else 0.0)
            feature_row.append(float(np.std(residuals[-self.lookback:])) if len(residuals) >= self.lookback else 0.0)
            feature_row.append(float(residuals[-1]))
        else:
            feature_row.extend([0.0, 0.0, 0.0])
        
        # Временные признаки
        if timestamps is not None:
            dt = timestamps
            if isinstance(dt, (datetime, pd.Timestamp)):
                feature_row.append(float(dt.hour))
                feature_row.append(float(dt.weekday()))
                feature_row.append(1.0 if dt.weekday() >= 5 else 0.0)
                feature_row.append(float(dt.month))
            else:
                feature_row.extend([0.0, 0.0, 0.0, 0.0])
        else:
            feature_row.extend([0.0, 0.0, 0.0, 0.0])
        
        # Масштабирование и предсказание
        try:
            features_scaled = self.scaler_X.transform([feature_row])
            prediction = self.rf_model.predict(features_scaled)[0]
            return float(prediction)
        except Exception as e:
            print(f"Ошибка RF прогноза: {e}")
            return None
    
    def predict(self, recent_values, timestamps=None, return_all=False):
        """Комбинированный прогноз"""
        # Прогноз SARIMA
        sarima_preds = self.predict_sarima()
        
        # Прогноз RF
        if self.ensemble_type == 'cascade' and self.rf_residual_model is not None:
            # Каскадный подход
            if self.sarima_results is not None and len(recent_values) > self.lookback:
                try:
                    hist_pred = self.sarima_results.predict(
                        start=len(recent_values) - self.lookback,
                        end=len(recent_values) - 1
                    )
                    if isinstance(hist_pred, pd.Series):
                        hist_pred = hist_pred.values
                    residuals = recent_values[-self.lookback:] - hist_pred
                except:
                    residuals = np.zeros(self.lookback)
            else:
                residuals = np.zeros(self.lookback)
            
            rf_pred = self.predict_random_forest(recent_values, timestamps, residuals)
            
            if sarima_preds is not None and rf_pred is not None:
                ensemble_preds = [sarima_preds[0] + rf_pred] + list(sarima_preds[1:])
            elif sarima_preds is not None:
                ensemble_preds = sarima_preds
            else:
                ensemble_preds = [rf_pred] * self.forecast_horizon
        else:
            # Взвешенное усреднение
            rf_pred = self.predict_random_forest(recent_values, timestamps)
            
            if sarima_preds is not None and rf_pred is not None:
                ensemble_preds = []
                for i in range(self.forecast_horizon):
                    if i == 0:
                        ensemble_val = (self.sarima_weight * sarima_preds[i] + 
                                       self.rf_weight * rf_pred)
                    else:
                        ensemble_val = sarima_preds[i]
                    
                    ensemble_val = max(0.0, min(100.0, ensemble_val))
                    ensemble_preds.append(ensemble_val)
            elif sarima_preds is not None:
                ensemble_preds = list(sarima_preds)
            elif rf_pred is not None:
                ensemble_preds = [rf_pred] * self.forecast_horizon
            else:
                return None
        
        ensemble_preds = np.array(ensemble_preds)
        
        if return_all:
            return {
                'sarima': sarima_preds,
                'random_forest': rf_pred,
                'ensemble': ensemble_preds
            }
        
        return ensemble_preds
    
    def _save_metadata(self):
        """Сохранение метаданных"""
        metadata = {
            'lookback': self.lookback,
            'forecast_horizon': self.forecast_horizon,
            'seasonal_period': self.seasonal_period,
            'ensemble_type': self.ensemble_type,
            'sarima_weight': self.sarima_weight,
            'rf_weight': self.rf_weight,
            'sarima_order': self.sarima_order,
            'sarima_seasonal_order': self.sarima_seasonal_order,
            'is_trained': self.is_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(self.model_dir, "RF_SARIMA_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        
        # Сохраняем модели
        if self.rf_model:
            joblib.dump(self.rf_model, os.path.join(self.model_dir, "RF_SARIMA_rf.pkl"))
            joblib.dump(self.scaler_X, os.path.join(self.model_dir, "RF_SARIMA_scaler.pkl"))
        
        if self.sarima_results:
            joblib.dump(self.sarima_results, os.path.join(self.model_dir, "SARIMA_results.pkl"))
        
        print(f"\n✓ Модели сохранены в: {self.model_dir}")
    
    def load_models(self):
        """Загрузка сохраненных моделей"""
        rf_path = os.path.join(self.model_dir, "RF_SARIMA_rf.pkl")
        if os.path.exists(rf_path):
            self.rf_model = joblib.load(rf_path)
            print("✓ Random Forest загружен")
        
        scaler_path = os.path.join(self.model_dir, "RF_SARIMA_scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler_X = joblib.load(scaler_path)
            print("✓ Скейлер загружен")
        
        sarima_path = os.path.join(self.model_dir, "SARIMA_results.pkl")
        if os.path.exists(sarima_path):
            self.sarima_results = joblib.load(sarima_path)
            print("✓ SARIMA загружена")
        
        metadata_path = os.path.join(self.model_dir, "RF_SARIMA_metadata.pkl")
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.lookback = metadata.get('lookback', self.lookback)
            self.forecast_horizon = metadata.get('forecast_horizon', self.forecast_horizon)
            self.is_trained = metadata.get('is_trained', False)
            print("✓ Метаданные загружены")
        
        return self.rf_model is not None or self.sarima_results is not None


class TimeSeriesDataLoader:
    """Загрузчик данных"""
    
    def __init__(self):
        self.data = None
    
    def load_from_csv(self, csv_path):
        """Загрузка из CSV"""
        try:
            self.data = pd.read_csv(csv_path)
            
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
        """Генерация синтетических данных"""
        if start_date is None:
            start_date = datetime(2024, 1, 1, 8, 0, 0)
        
        timestamps = [start_date + timedelta(minutes=10 * i) for i in range(n_points)]
        
        np.random.seed(42)
        
        hours = np.array([ts.hour + ts.minute/60 for ts in timestamps])
        seasonal = 30 + 40 * np.sin(2 * np.pi * hours / 24)
        trend = 5 * np.sin(np.linspace(0, np.pi, n_points))
        noise = np.random.normal(0, 5, n_points)
        
        series = seasonal + trend + noise
        for i in range(1, n_points):
            series[i] = 0.8 * series[i] + 0.2 * series[i-1]
        
        occupancy = np.clip(series, 0, 100)
        
        return timestamps, occupancy


class EnsembleVisualizer:
    """Визуализатор"""
    
    def __init__(self, ensemble_model):
        self.ensemble = ensemble_model
    
    def plot_sarima_diagnostics(self):
        """Диагностика SARIMA"""
        if self.ensemble.sarima_results is None:
            print("SARIMA модель не обучена")
            return
        
        fig = self.ensemble.sarima_results.plot_diagnostics(figsize=(12, 8))
        fig.suptitle('Диагностика SARIMA модели', fontsize=14)
        
        save_path = os.path.join(config.RESULTS_DIR, 'sarima_diagnostics.png')
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nГрафик сохранен: {save_path}")
        plt.show()
    
    def plot_predictions(self, actual_values, predictions, title="Прогноз загрузки парковки"):
        """Визуализация прогнозов"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Основной прогноз
        ax1 = axes[0, 0]
        ax1.plot(range(len(actual_values)), actual_values, 'b-', label='Фактические', linewidth=2)
        
        pred_start = len(actual_values) - 1
        pred_x = list(range(pred_start, pred_start + len(predictions) + 1))
        pred_y = [actual_values[-1]] + list(predictions)
        ax1.plot(pred_x, pred_y, 'r--', label='Прогноз', linewidth=2, marker='o')
        ax1.set_title(title, fontsize=12)
        ax1.set_xlabel('Временной шаг')
        ax1.set_ylabel('Загрузка (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Состав ансамбля
        ax2 = axes[0, 1]
        ax2.text(0.5, 0.6, f'SARIMA\nВес: {self.ensemble.sarima_weight}', 
                ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightcoral'))
        ax2.text(0.5, 0.3, f'Random Forest\nВес: {self.ensemble.rf_weight}', 
                ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue'))
        ax2.set_title('Состав ансамбля', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        # Важность признаков
        ax3 = axes[1, 0]
        if hasattr(self.ensemble, 'rf_model') and self.ensemble.rf_model:
            importances = self.ensemble.rf_model.feature_importances_
            n_features = min(15, len(importances))
            indices = np.argsort(importances)[-n_features:]
            
            ax3.barh(range(n_features), importances[indices])
            ax3.set_yticks(range(n_features))
            ax3.set_yticklabels([f'F{i}' for i in indices])
            ax3.set_xlabel('Важность')
            ax3.set_title('Важность признаков (RF)', fontsize=12)
        else:
            ax3.text(0.5, 0.5, 'RF модель не обучена', ha='center', va='center')
        
        # Остатки
        ax4 = axes[1, 1]
        if self.ensemble.sarima_results is not None:
            residuals = self.ensemble.sarima_results.resid
            if len(residuals) > 100:
                residuals = residuals[-100:]
            ax4.plot(residuals, 'b-', alpha=0.7)
            ax4.axhline(y=0, color='r', linestyle='--')
            ax4.set_title('Остатки SARIMA', fontsize=12)
            ax4.set_xlabel('Временной шаг')
            ax4.set_ylabel('Остатки')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'SARIMA не обучена', ha='center', va='center')
        
        plt.tight_layout()
        
        save_path = os.path.join(config.RESULTS_DIR, 'rf_sarima_predictions.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nГрафик сохранен: {save_path}")
        plt.show()
    
    def plot_forecast_comparison(self, sarima_pred, rf_pred, ensemble_pred):
        """Сравнение прогнозов"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        steps = range(len(ensemble_pred))
        
        if sarima_pred is not None:
            ax.plot(steps, sarima_pred, 'g--', label='SARIMA', linewidth=2, marker='s', markersize=4)
        
        if rf_pred is not None:
            ax.axhline(y=rf_pred, color='orange', linestyle='--', 
                      label=f'RF: {rf_pred:.1f}%', linewidth=2)
        
        ax.plot(steps, ensemble_pred, 'r-', label='Ансамбль RF+SARIMA', linewidth=2, marker='o', markersize=6)
        
        ax.set_xlabel('Шаг прогноза', fontsize=12)
        ax.set_ylabel('Загрузка (%)', fontsize=12)
        ax.set_title('Сравнение прогнозов', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        ax.text(0.02, 0.98, f'Тип ансамбля: {self.ensemble.ensemble_type}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        save_path = os.path.join(config.RESULTS_DIR, 'rf_sarima_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранен: {save_path}")
        plt.show()


def main():
    """Главная функция"""
    
    print("\n" + "="*70)
    print("КОМБИНИРОВАННАЯ МОДЕЛЬ RF + SARIMA")
    print("="*70)
    
    # Параметры
    LOOKBACK = 24
    FORECAST_HORIZON = 12
    SEASONAL_PERIOD = 24
    ENSEMBLE_TYPE = 'weighted'
    
    print(f"\nПараметры:")
    print(f"  Lookback: {LOOKBACK} шагов")
    print(f"  Горизонт: {FORECAST_HORIZON} шагов")
    print(f"  Сезонный период: {SEASONAL_PERIOD}")
    print(f"  Тип ансамбля: {ENSEMBLE_TYPE}")
    
    # Загрузка данных
    loader = TimeSeriesDataLoader()
    
    data_path = os.path.join(config.RESULTS_DIR, "parking_occupancy_full.csv")
    timestamps, occupancy = None, None
    
    if os.path.exists(data_path):
        print(f"\nЗагрузка из: {data_path}")
        timestamps, occupancy = loader.load_from_csv(data_path)
        if occupancy is not None:
            print(f"  Загружено {len(occupancy)} записей")
    
    if occupancy is None or len(occupancy) < 100:
        print("\nГенерация синтетических данных...")
        timestamps, occupancy = loader.generate_synthetic_data(n_points=500)
        print(f"  Сгенерировано {len(occupancy)} записей")
    
    # Создание и обучение модели
    ensemble = EnsembleRFSARIMA(
        lookback=LOOKBACK,
        forecast_horizon=FORECAST_HORIZON,
        seasonal_period=SEASONAL_PERIOD,
        ensemble_type=ENSEMBLE_TYPE
    )
    
    print("\n" + "="*70)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("="*70)
    
    success = ensemble.train(occupancy, timestamps, optimize_sarima=False)
    
    if success:
        print("\n✅ Модель обучена!")
        
        # Тестирование
        print("\n" + "="*70)
        print("ПРОГНОЗ")
        print("="*70)
        
        recent_values = occupancy[-LOOKBACK:]
        current_time = timestamps[-1] if timestamps else None
        
        predictions = ensemble.predict(recent_values, current_time, return_all=True)
        
        if predictions:
            print(f"\nПрогноз на {FORECAST_HORIZON} шагов:")
            print("-" * 60)
            print(f"{'Шаг':<6} {'SARIMA':<12} {'RF':<12} {'Ансамбль':<12}")
            print("-" * 60)
            
            for i in range(FORECAST_HORIZON):
                sarima = f"{predictions['sarima'][i]:.1f}%" if predictions['sarima'] is not None else "-"
                rf = f"{predictions['random_forest']:.1f}%" if i == 0 and predictions['random_forest'] else "-"
                ensemble_val = f"{predictions['ensemble'][i]:.1f}%"
                print(f"{i+1:<6} {sarima:<12} {rf:<12} {ensemble_val:<12}")
            
            print("-" * 60)
            
            # Визуализация
            visualizer = EnsembleVisualizer(ensemble)
            visualizer.plot_sarima_diagnostics()
            visualizer.plot_predictions(occupancy[-50:], predictions['ensemble'])
            visualizer.plot_forecast_comparison(
                predictions['sarima'],
                predictions['random_forest'],
                predictions['ensemble']
            )
            
            # Рекомендации
            print("\n" + "="*70)
            print("РЕКОМЕНДАЦИИ")
            print("="*70)
            
            current = occupancy[-1]
            next_val = predictions['ensemble'][0]
            
            print(f"\nТекущая загрузка: {current:.1f}%")
            print(f"Прогноз: {next_val:.1f}%")
            
            if next_val < current - 5:
                print("\n📉 Загрузка снизится - можно подождать")
            elif next_val > current + 5:
                print("\n📈 Загрузка вырастет - паркуйтесь сейчас")
            else:
                print("\n➡ Загрузка стабильна")
    
    else:
        print("\n❌ Ошибка обучения")
    
    print("\n" + "="*70)
    print("ЗАВЕРШЕНО")
    print("="*70)


if __name__ == "__main__":
    main()