# compare_ensembles.py
"""
Сравнение комбинированных моделей прогнозирования:
1. Random Forest + LSTM
2. Random Forest + SARIMA

Метрики сравнения:
- MAE, MSE, RMSE
- MAPE, SMAPE, WAPE
- R² Score, Bias, Max Error
- Время обучения и предсказания
- Точность в пределах 10% и 20%
"""

import numpy as np
import pandas as pd
import os
import time
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import config

# Импорт моделей
import sys
sys.path.append(os.path.dirname(__file__))

from ensemble_rf_lstm import EnsembleRFLSTM
from ensemble_rf_sarima import EnsembleRFSARIMA


class ModelComparator:
    """Класс для сравнения моделей"""
    
    def __init__(self, test_size=0.2, lookback=24, forecast_horizon=12):
        self.test_size = test_size
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.results = {}
        
    def prepare_data(self, time_series, timestamps=None):
        """Подготовка данных для тестирования"""
        if isinstance(time_series, pd.Series):
            time_series = time_series.values
        
        split_idx = int(len(time_series) * (1 - self.test_size))
        
        train_series = time_series[:split_idx]
        test_series = time_series[split_idx:]
        
        train_timestamps = timestamps[:split_idx] if timestamps is not None else None
        test_timestamps = timestamps[split_idx:] if timestamps is not None else None
        
        print(f"\nРазделение данных:")
        print(f"  Train: {len(train_series)} записей")
        print(f"  Test: {len(test_series)} записей")
        
        return train_series, test_series, train_timestamps, test_timestamps
    
    def train_and_predict_rf_lstm(self, train_series, test_series, train_timestamps=None, test_timestamps=None):
        """Обучение и прогноз RF+LSTM модели"""
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ МОДЕЛИ RF + LSTM")
        print("="*60)
        
        start_time = time.time()
        
        model = EnsembleRFLSTM(
            lookback=self.lookback,
            forecast_horizon=self.forecast_horizon,
            model_dir=os.path.join(config.MODELS_DIR, "compare_rf_lstm")
        )
        
        success = model.train(train_series, train_timestamps)
        train_time = time.time() - start_time
        
        if not success:
            print("❌ Ошибка обучения RF+LSTM")
            return None, None, None
        
        print("\nПрогнозирование на тестовых данных...")
        predictions = []
        actuals = []
        
        start_pred_time = time.time()
        
        for i in range(len(test_series) - self.forecast_horizon):
            if i >= self.lookback:
                recent = test_series[i - self.lookback:i]
            else:
                recent = np.concatenate([train_series[-self.lookback + i:], test_series[:i]])
            
            current_time = test_timestamps[i] if test_timestamps is not None else None
            pred = model.predict(recent, current_time, return_all=False)
            
            if pred is not None:
                predictions.append(pred[0])
                actuals.append(test_series[i + 1])
        
        pred_time = time.time() - start_pred_time
        
        if len(predictions) == 0:
            print("❌ Не удалось сделать прогнозы")
            return None, None, None
        
        metrics = self.calculate_metrics(actuals, predictions)
        metrics['train_time'] = train_time
        metrics['pred_time'] = pred_time
        metrics['predictions'] = predictions
        metrics['actuals'] = actuals
        
        print(f"\n✅ RF+LSTM результаты:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        return model, predictions, metrics
    
    def train_and_predict_rf_sarima(self, train_series, test_series, train_timestamps=None, test_timestamps=None):
        """Обучение и прогноз RF+SARIMA модели"""
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ МОДЕЛИ RF + SARIMA")
        print("="*60)
        
        start_time = time.time()
        
        model = EnsembleRFSARIMA(
            lookback=self.lookback,
            forecast_horizon=self.forecast_horizon,
            seasonal_period=24,
            model_dir=os.path.join(config.MODELS_DIR, "compare_rf_sarima"),
            ensemble_type='weighted'
        )
        
        success = model.train(train_series, train_timestamps, optimize_sarima=False)
        train_time = time.time() - start_time
        
        if not success:
            print("❌ Ошибка обучения RF+SARIMA")
            return None, None, None
        
        print("\nПрогнозирование на тестовых данных...")
        predictions = []
        actuals = []
        
        start_pred_time = time.time()
        
        for i in range(len(test_series) - self.forecast_horizon):
            if i >= self.lookback:
                recent = test_series[i - self.lookback:i]
            else:
                recent = np.concatenate([train_series[-self.lookback + i:], test_series[:i]])
            
            current_time = test_timestamps[i] if test_timestamps is not None else None
            pred = model.predict(recent, current_time, return_all=False)
            
            if pred is not None:
                predictions.append(pred[0])
                actuals.append(test_series[i + 1])
        
        pred_time = time.time() - start_pred_time
        
        if len(predictions) == 0:
            print("❌ Не удалось сделать прогнозы")
            return None, None, None
        
        metrics = self.calculate_metrics(actuals, predictions)
        metrics['train_time'] = train_time
        metrics['pred_time'] = pred_time
        metrics['predictions'] = predictions
        metrics['actuals'] = actuals
        
        print(f"\n✅ RF+SARIMA результаты:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        return model, predictions, metrics
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Расчет метрик качества
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # MAE, MSE, RMSE
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # ========== MAPE (с защитой от выбросов) ==========
        # Игнорируем точки, где фактическое значение < 1%
        MAPE_THRESHOLD = 1.0
        
        mask = y_true >= MAPE_THRESHOLD
        
        if np.sum(mask) > 0:
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
        else:
            mape = float('inf')
        
        # ========== SMAPE ==========
        # Используем более стабильную формулу с делением на 2
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        denominator = np.where(denominator < 0.5, 0.5, denominator)  # Защита от слишком малых знаменателей
        smape = np.mean(100 * np.abs(y_pred - y_true) / denominator)
        
        # ========== WAPE (взвешенный) ==========
        wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
        
        # ========== R² ==========
        r2 = r2_score(y_true, y_pred)
        
        # ========== Max Error ==========
        absolute_errors = np.abs(y_true - y_pred)
        max_error = np.max(absolute_errors)
        
        # ========== Bias ==========
        bias = np.mean(y_pred - y_true)
        
        # ========== Точность в пределах 10% и 20% ==========
        y_true_safe = np.where(y_true < 1.0, 1.0, y_true)
        within_10pct = np.mean(absolute_errors / y_true_safe <= 0.1) * 100
        within_20pct = np.mean(absolute_errors / y_true_safe <= 0.2) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE (%)': mape,
            'SMAPE (%)': smape,
            'WAPE (%)': wape,
            'R²': r2,
            'Max Error': max_error,
            'Bias': bias,
            'Within 10% (%)': within_10pct,
            'Within 20% (%)': within_20pct
        }
    
    def compare(self, time_series, timestamps=None):
        """Полное сравнение моделей"""
        print("\n" + "="*70)
        print("СРАВНЕНИЕ МОДЕЛЕЙ ПРОГНОЗИРОВАНИЯ")
        print("RF+LSTM vs RF+SARIMA")
        print("="*70)
        
        train_series, test_series, train_ts, test_ts = self.prepare_data(time_series, timestamps)
        
        _, preds_lstm, metrics_lstm = self.train_and_predict_rf_lstm(
            train_series, test_series, train_ts, test_ts
        )
        
        _, preds_sarima, metrics_sarima = self.train_and_predict_rf_sarima(
            train_series, test_series, train_ts, test_ts
        )
        
        if metrics_lstm and metrics_sarima:
            winner = self.determine_winner(metrics_lstm, metrics_sarima)
            self.results = {
                'RF_LSTM': metrics_lstm,
                'RF_SARIMA': metrics_sarima,
                'winner': winner
            }
            return self.results
        
        return None
    
    def determine_winner(self, metrics_lstm, metrics_sarima):
        """Определение лучшей модели по всем метрикам"""
        scores = {'RF+LSTM': 0, 'RF+SARIMA': 0}
        
        # MAE (меньше лучше)
        if metrics_lstm['MAE'] < metrics_sarima['MAE']:
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        # MSE (меньше лучше)
        if metrics_lstm['MSE'] < metrics_sarima['MSE']:
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        # RMSE (меньше лучше)
        if metrics_lstm['RMSE'] < metrics_sarima['RMSE']:
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        # MAPE (меньше лучше)
        if metrics_lstm['MAPE (%)'] < metrics_sarima['MAPE (%)']:
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        # SMAPE (меньше лучше)
        if metrics_lstm['SMAPE (%)'] < metrics_sarima['SMAPE (%)']:
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        # WAPE (меньше лучше)
        if metrics_lstm['WAPE (%)'] < metrics_sarima['WAPE (%)']:
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        # R² (больше лучше)
        if metrics_lstm['R²'] > metrics_sarima['R²']:
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        # Max Error (меньше лучше)
        if metrics_lstm['Max Error'] < metrics_sarima['Max Error']:
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        # Bias (ближе к нулю лучше)
        if abs(metrics_lstm['Bias']) < abs(metrics_sarima['Bias']):
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        # Within 10% (больше лучше)
        if metrics_lstm['Within 10% (%)'] > metrics_sarima['Within 10% (%)']:
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        # Within 20% (больше лучше)
        if metrics_lstm['Within 20% (%)'] > metrics_sarima['Within 20% (%)']:
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        # Время обучения (меньше лучше)
        if metrics_lstm['train_time'] < metrics_sarima['train_time']:
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        # Время прогноза (меньше лучше)
        if metrics_lstm['pred_time'] < metrics_sarima['pred_time']:
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        if scores['RF+LSTM'] > scores['RF+SARIMA']:
            return 'RF+LSTM'
        elif scores['RF+SARIMA'] > scores['RF+LSTM']:
            return 'RF+SARIMA'
        else:
            return 'Ничья'


class ResultsVisualizer:
    """Класс для визуализации результатов отдельными графиками"""
    
    def __init__(self, results):
        self.results = results
    
    def _is_better(self, metric_name, val_lstm, val_sarima):
        """Определяет, какая модель лучше по данной метрике"""
        lower_is_better = ['MAE', 'MSE', 'RMSE', 'MAPE (%)', 'SMAPE (%)', 
                          'WAPE (%)', 'Max Error', 'train_time', 'pred_time']
        higher_is_better = ['R²', 'Within 10% (%)', 'Within 20% (%)']
        
        if metric_name == 'Bias':
            if abs(val_lstm) < abs(val_sarima):
                return 'lstm'
            elif abs(val_sarima) < abs(val_lstm):
                return 'sarima'
            return 'tie'
        
        if metric_name in lower_is_better:
            if val_lstm < val_sarima:
                return 'lstm'
            elif val_sarima < val_lstm:
                return 'sarima'
            return 'tie'
        elif metric_name in higher_is_better:
            if val_lstm > val_sarima:
                return 'lstm'
            elif val_sarima > val_lstm:
                return 'sarima'
            return 'tie'
        
        return 'tie'
    
    def plot_comparison_table(self):
        """График 1: Таблица сравнения метрик"""
        if not self.results:
            print("Нет данных для отображения")
            return
        
        metrics_lstm = self.results['RF_LSTM']
        metrics_sarima = self.results['RF_SARIMA']
        winner = self.results['winner']
        
        fig, ax = plt.subplots(figsize=(16, 14))
        ax.axis('tight')
        ax.axis('off')
        
        # Определяем лучшие значения для каждой метрики
        best_model = {}
        for metric in metrics_lstm.keys():
            if metric not in ['predictions', 'actuals']:
                best_model[metric] = self._is_better(metric, metrics_lstm[metric], metrics_sarima[metric])
        
        table_data = [
            ('📊 ОСНОВНЫЕ МЕТРИКИ ОШИБОК', '', ''),
            ('MAE (средняя абсолютная ошибка)', f"{metrics_lstm['MAE']:.4f}", f"{metrics_sarima['MAE']:.4f}"),
            ('MSE (среднеквадратичная ошибка)', f"{metrics_lstm['MSE']:.4f}", f"{metrics_sarima['MSE']:.4f}"),
            ('RMSE (корень из MSE)', f"{metrics_lstm['RMSE']:.4f}", f"{metrics_sarima['RMSE']:.4f}"),
            ('', '', ''),
            ('📈 ПРОЦЕНТНЫЕ МЕТРИКИ ОШИБОК', '', ''),
            ('MAPE (%)', f"{metrics_lstm['MAPE (%)']:.2f}%", f"{metrics_sarima['MAPE (%)']:.2f}%"),
            ('SMAPE (%)', f"{metrics_lstm['SMAPE (%)']:.2f}%", f"{metrics_sarima['SMAPE (%)']:.2f}%"),
            ('WAPE (%)', f"{metrics_lstm['WAPE (%)']:.2f}%", f"{metrics_sarima['WAPE (%)']:.2f}%"),
            ('', '', ''),
            ('🎯 КАЧЕСТВО МОДЕЛИ', '', ''),
            ('R² (коэф. детерминации)', f"{metrics_lstm['R²']:.4f}", f"{metrics_sarima['R²']:.4f}"),
            ('Max Error (максимальная ошибка)', f"{metrics_lstm['Max Error']:.4f}", f"{metrics_sarima['Max Error']:.4f}"),
            ('Bias (систематическое смещение)', f"{metrics_lstm['Bias']:.4f}", f"{metrics_sarima['Bias']:.4f}"),
            ('', '', ''),
            ('✅ ТОЧНОСТЬ ПРОГНОЗОВ', '', ''),
            ('Within 10% (ошибка ≤10%)', f"{metrics_lstm['Within 10% (%)']:.1f}%", f"{metrics_sarima['Within 10% (%)']:.1f}%"),
            ('Within 20% (ошибка ≤20%)', f"{metrics_lstm['Within 20% (%)']:.1f}%", f"{metrics_sarima['Within 20% (%)']:.1f}%"),
            ('', '', ''),
            ('⚡ ПРОИЗВОДИТЕЛЬНОСТЬ', '', ''),
            ('Время обучения (сек)', f"{metrics_lstm['train_time']:.2f}", f"{metrics_sarima['train_time']:.2f}"),
            ('Время прогноза (сек)', f"{metrics_lstm['pred_time']:.2f}", f"{metrics_sarima['pred_time']:.2f}"),
        ]
        
        col_labels = ['Метрика', '🤖 RF+LSTM', '📈 RF+SARIMA']
        
        table = ax.table(cellText=table_data, colLabels=col_labels,
                        cellLoc='left', loc='center', colWidths=[0.45, 0.25, 0.25])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.6)
        
        metric_mapping = {
            'MAE': 'MAE', 'MSE': 'MSE', 'RMSE': 'RMSE',
            'MAPE': 'MAPE (%)', 'SMAPE': 'SMAPE (%)', 'WAPE': 'WAPE (%)',
            'R²': 'R²', 'Max Error': 'Max Error', 'Bias': 'Bias',
            'Within 10%': 'Within 10% (%)', 'Within 20%': 'Within 20% (%)',
            'Время обучения': 'train_time', 'Время прогноза': 'pred_time'
        }
        
        for i in range(len(table_data) + 1):
            for j in range(3):
                cell = table[(i, j)]
                
                if i == 0:
                    cell.set_facecolor('#2C3E50')
                    cell.set_text_props(weight='bold', color='white', fontsize=11)
                else:
                    if i % 2 == 0:
                        cell.set_facecolor('#F8F9F9')
                    else:
                        cell.set_facecolor('#ECF0F1')
                    
                    text = cell.get_text().get_text()
                    
                    if text in ['📊 ОСНОВНЫЕ МЕТРИКИ ОШИБОК', '📈 ПРОЦЕНТНЫЕ МЕТРИКИ ОШИБОК',
                               '🎯 КАЧЕСТВО МОДЕЛИ', '✅ ТОЧНОСТЬ ПРОГНОЗОВ', '⚡ ПРОИЗВОДИТЕЛЬНОСТЬ']:
                        cell.set_text_props(weight='bold', fontsize=10, color='#2980B9')
                    
                    elif j == 1:
                        for metric_ru, metric_key in metric_mapping.items():
                            if metric_ru in table_data[i-1][0]:
                                if best_model.get(metric_key) == 'lstm':
                                    cell.set_text_props(weight='bold', color='#27AE60', fontsize=10)
                                break
                    
                    elif j == 2:
                        for metric_ru, metric_key in metric_mapping.items():
                            if metric_ru in table_data[i-1][0]:
                                if best_model.get(metric_key) == 'sarima':
                                    cell.set_text_props(weight='bold', color='#27AE60', fontsize=10)
                                break
        
        ax.set_title(f'СРАВНЕНИЕ МОДЕЛЕЙ ПРОГНОЗИРОВАНИЯ\n🏆 Победитель: {winner} 🏆', 
                    fontsize=16, fontweight='bold', pad=30)
        
        fig.text(0.5, 0.02, '✅ Зеленым цветом выделены лучшие показатели по каждой метрике',
                ha='center', fontsize=9, style='italic', color='#7F8C8D')
        
        plt.tight_layout()
        
        save_path = os.path.join(config.RESULTS_DIR, '01_comparison_table.png')
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\n📊 График 1 сохранен: {save_path}")
        plt.show()
        
        return fig
    
    def plot_predictions_comparison(self):
        """График 2: Сравнение прогнозов обеих моделей"""
        metrics_lstm = self.results['RF_LSTM']
        metrics_sarima = self.results['RF_SARIMA']
        
        actuals = np.array(metrics_lstm['actuals'])
        preds_lstm = np.array(metrics_lstm['predictions'])
        preds_sarima = np.array(metrics_sarima['predictions'])
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # График 1: RF+LSTM
        ax1 = axes[0]
        ax1.plot(actuals, label='Фактические значения', color='#2C3E50', linewidth=1.5, alpha=0.8)
        ax1.plot(preds_lstm, label='Прогноз RF+LSTM', color='#E74C3C', linewidth=1.5, alpha=0.8)
        ax1.fill_between(range(len(actuals)), actuals, preds_lstm, alpha=0.2, color='#E74C3C')
        ax1.set_title(f'RF+LSTM: Сравнение прогнозов\nMAE={metrics_lstm["MAE"]:.3f}, SMAPE={metrics_lstm["SMAPE (%)"]:.2f}%, R²={metrics_lstm["R²"]:.4f}', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Временной шаг')
        ax1.set_ylabel('Значение')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # График 2: RF+SARIMA
        ax2 = axes[1]
        ax2.plot(actuals, label='Фактические значения', color='#2C3E50', linewidth=1.5, alpha=0.8)
        ax2.plot(preds_sarima, label='Прогноз RF+SARIMA', color='#3498DB', linewidth=1.5, alpha=0.8)
        ax2.fill_between(range(len(actuals)), actuals, preds_sarima, alpha=0.2, color='#3498DB')
        ax2.set_title(f'RF+SARIMA: Сравнение прогнозов\nMAE={metrics_sarima["MAE"]:.3f}, SMAPE={metrics_sarima["SMAPE (%)"]:.2f}%, R²={metrics_sarima["R²"]:.4f}', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Временной шаг')
        ax2.set_ylabel('Значение')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(config.RESULTS_DIR, '02_predictions_comparison.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"📈 График 2 сохранен: {save_path}")
        plt.show()
        
        return fig
    
    def plot_errors_comparison(self):
        """График 3: Сравнение ошибок прогнозов"""
        metrics_lstm = self.results['RF_LSTM']
        metrics_sarima = self.results['RF_SARIMA']
        
        actuals = np.array(metrics_lstm['actuals'])
        preds_lstm = np.array(metrics_lstm['predictions'])
        preds_sarima = np.array(metrics_sarima['predictions'])
        
        errors_lstm = preds_lstm - actuals
        errors_sarima = preds_sarima - actuals
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Гистограммы ошибок
        axes[0, 0].hist(errors_lstm, bins=30, alpha=0.7, color='#E74C3C', edgecolor='black')
        axes[0, 0].axvline(x=0, color='black', linestyle='--', linewidth=1)
        axes[0, 0].axvline(x=np.mean(errors_lstm), color='#E74C3C', linestyle='-', linewidth=2, 
                          label=f'Среднее: {np.mean(errors_lstm):.3f}')
        axes[0, 0].set_title(f'RF+LSTM: Распределение ошибок\nСтд. отклонение: {np.std(errors_lstm):.3f}', fontsize=11)
        axes[0, 0].set_xlabel('Ошибка прогноза')
        axes[0, 0].set_ylabel('Частота')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(errors_sarima, bins=30, alpha=0.7, color='#3498DB', edgecolor='black')
        axes[0, 1].axvline(x=0, color='black', linestyle='--', linewidth=1)
        axes[0, 1].axvline(x=np.mean(errors_sarima), color='#3498DB', linestyle='-', linewidth=2,
                          label=f'Среднее: {np.mean(errors_sarima):.3f}')
        axes[0, 1].set_title(f'RF+SARIMA: Распределение ошибок\nСтд. отклонение: {np.std(errors_sarima):.3f}', fontsize=11)
        axes[0, 1].set_xlabel('Ошибка прогноза')
        axes[0, 1].set_ylabel('Частота')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot сравнения
        bp_data = [errors_lstm, errors_sarima]
        bp = axes[1, 0].boxplot(bp_data, labels=['RF+LSTM', 'RF+SARIMA'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#E74C3C')
        bp['boxes'][1].set_facecolor('#3498DB')
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1, 0].set_title('Сравнение распределения ошибок', fontsize=11)
        axes[1, 0].set_ylabel('Ошибка прогноза')
        axes[1, 0].grid(True, alpha=0.3)
        
        # График ошибок во времени
        axes[1, 1].plot(errors_lstm, label='RF+LSTM', color='#E74C3C', alpha=0.7, linewidth=1)
        axes[1, 1].plot(errors_sarima, label='RF+SARIMA', color='#3498DB', alpha=0.7, linewidth=1)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1, 1].set_title('Ошибки прогнозов во времени', fontsize=11)
        axes[1, 1].set_xlabel('Временной шаг')
        axes[1, 1].set_ylabel('Ошибка')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(config.RESULTS_DIR, '03_errors_comparison.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"📉 График 3 сохранен: {save_path}")
        plt.show()
        
        return fig
    
    def plot_metrics_bars(self):
        """График 4: Сравнение ключевых метрик (столбчатая диаграмма)"""
        metrics_lstm = self.results['RF_LSTM']
        metrics_sarima = self.results['RF_SARIMA']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # MAE, RMSE сравнение
        metrics_names = ['MAE', 'RMSE']
        lstm_values = [metrics_lstm['MAE'], metrics_lstm['RMSE']]
        sarima_values = [metrics_sarima['MAE'], metrics_sarima['RMSE']]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, lstm_values, width, label='RF+LSTM', color='#E74C3C')
        axes[0, 0].bar(x + width/2, sarima_values, width, label='RF+SARIMA', color='#3498DB')
        axes[0, 0].set_title('Сравнение ошибок (меньше = лучше)', fontsize=11, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics_names)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Процентные ошибки
        pct_names = ['MAPE', 'SMAPE', 'WAPE']
        lstm_pct = [metrics_lstm['MAPE (%)'], metrics_lstm['SMAPE (%)'], metrics_lstm['WAPE (%)']]
        sarima_pct = [metrics_sarima['MAPE (%)'], metrics_sarima['SMAPE (%)'], metrics_sarima['WAPE (%)']]
        
        x = np.arange(len(pct_names))
        axes[0, 1].bar(x - width/2, lstm_pct, width, label='RF+LSTM', color='#E74C3C')
        axes[0, 1].bar(x + width/2, sarima_pct, width, label='RF+SARIMA', color='#3498DB')
        axes[0, 1].set_title('Процентные ошибки (меньше = лучше)', fontsize=11, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(pct_names)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # R² и точность
        quality_names = ['R²', 'Within 10%', 'Within 20%']
        lstm_quality = [metrics_lstm['R²'], metrics_lstm['Within 10% (%)'], metrics_lstm['Within 20% (%)']]
        sarima_quality = [metrics_sarima['R²'], metrics_sarima['Within 10% (%)'], metrics_sarima['Within 20% (%)']]
        
        x = np.arange(len(quality_names))
        axes[1, 0].bar(x - width/2, lstm_quality, width, label='RF+LSTM', color='#E74C3C')
        axes[1, 0].bar(x + width/2, sarima_quality, width, label='RF+SARIMA', color='#3498DB')
        axes[1, 0].set_title('Качество модели (больше = лучше)', fontsize=11, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(quality_names)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Время выполнения
        time_names = ['Training', 'Prediction']
        lstm_time = [metrics_lstm['train_time'], metrics_lstm['pred_time']]
        sarima_time = [metrics_sarima['train_time'], metrics_sarima['pred_time']]
        
        x = np.arange(len(time_names))
        axes[1, 1].bar(x - width/2, lstm_time, width, label='RF+LSTM', color='#E74C3C')
        axes[1, 1].bar(x + width/2, sarima_time, width, label='RF+SARIMA', color='#3498DB')
        axes[1, 1].set_title('Время выполнения (сек, меньше = лучше)', fontsize=11, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(time_names)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        save_path = os.path.join(config.RESULTS_DIR, '04_metrics_bars.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"📊 График 4 сохранен: {save_path}")
        plt.show()
        
        return fig
    
    def plot_scatter_comparison(self):
        """График 5: Диаграммы рассеяния (фактические vs прогнозы)"""
        metrics_lstm = self.results['RF_LSTM']
        metrics_sarima = self.results['RF_SARIMA']
        
        actuals = np.array(metrics_lstm['actuals'])
        preds_lstm = np.array(metrics_lstm['predictions'])
        preds_sarima = np.array(metrics_sarima['predictions'])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # RF+LSTM scatter
        ax1 = axes[0]
        ax1.scatter(actuals, preds_lstm, alpha=0.6, color='#E74C3C', s=30)
        
        # Линия идеального прогноза
        min_val = min(actuals.min(), preds_lstm.min())
        max_val = max(actuals.max(), preds_lstm.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Идеальный прогноз')
        
        # Линия регрессии
        z = np.polyfit(actuals, preds_lstm, 1)
        p = np.poly1d(z)
        ax1.plot(actuals, p(actuals), 'r-', alpha=0.5, label=f'Тренд (y={z[0]:.2f}x+{z[1]:.2f})')
        
        ax1.set_xlabel('Фактические значения')
        ax1.set_ylabel('Прогнозные значения')
        ax1.set_title(f'RF+LSTM\nR² = {metrics_lstm["R²"]:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RF+SARIMA scatter
        ax2 = axes[1]
        ax2.scatter(actuals, preds_sarima, alpha=0.6, color='#3498DB', s=30)
        
        min_val = min(actuals.min(), preds_sarima.min())
        max_val = max(actuals.max(), preds_sarima.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Идеальный прогноз')
        
        z = np.polyfit(actuals, preds_sarima, 1)
        p = np.poly1d(z)
        ax2.plot(actuals, p(actuals), 'b-', alpha=0.5, label=f'Тренд (y={z[0]:.2f}x+{z[1]:.2f})')
        
        ax2.set_xlabel('Фактические значения')
        ax2.set_ylabel('Прогнозные значения')
        ax2.set_title(f'RF+SARIMA\nR² = {metrics_sarima["R²"]:.4f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(config.RESULTS_DIR, '05_scatter_comparison.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"🎯 График 5 сохранен: {save_path}")
        plt.show()
        
        return fig
    
    def plot_all(self):
        """Вывод всех графиков"""
        print("\n" + "="*60)
        print("ГЕНЕРАЦИЯ ГРАФИКОВ СРАВНЕНИЯ")
        print("="*60)
        
        self.plot_comparison_table()
        self.plot_predictions_comparison()
        self.plot_errors_comparison()
        self.plot_metrics_bars()
        self.plot_scatter_comparison()
        
        print("\n✅ Все графики успешно сгенерированы!")


def generate_synthetic_data(n_points=500, pattern='mixed'):
    """Генерация синтетических данных"""
    np.random.seed(42)
    
    timestamps = [datetime(2024, 1, 1, 8, 0, 0) + timedelta(minutes=10 * i) for i in range(n_points)]
    
    hours = np.array([ts.hour + ts.minute/60 for ts in timestamps])
    seasonal = 30 + 35 * np.sin(2 * np.pi * hours / 24)
    trend = 5 * np.sin(np.linspace(0, np.pi, n_points))
    noise = np.random.normal(0, 5, n_points)
    occupancy = seasonal + trend + noise
    occupancy = np.clip(occupancy, 0, 100)
    
    return timestamps, occupancy


def main():
    """Главная функция"""
    
    print("\n" + "="*70)
    print("СРАВНЕНИЕ МОДЕЛЕЙ ПРОГНОЗИРОВАНИЯ")
    print("RF+LSTM vs RF+SARIMA")
    print("="*70)
    
    LOOKBACK = 24
    FORECAST_HORIZON = 12
    TEST_SIZE = 0.2
    
    print(f"\nПараметры сравнения:")
    print(f"  Lookback: {LOOKBACK} шагов")
    print(f"  Горизонт прогноза: {FORECAST_HORIZON} шагов")
    print(f"  Тестовый размер: {TEST_SIZE * 100}%")
    
    # Загрузка данных
    data_path = os.path.join(config.RESULTS_DIR, "parking_occupancy_full.csv")
    
    timestamps, occupancy = None, None
    
    if os.path.exists(data_path):
        print(f"\nЗагрузка реальных данных...")
        try:
            df = pd.read_csv(data_path)
            if 'timestamp' in df.columns:
                timestamps = pd.to_datetime(df['timestamp']).values
            if 'occupancy_rate' in df.columns:
                occupancy = df['occupancy_rate'].values
            print(f"✓ Загружено {len(occupancy)} записей")
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
    
    if occupancy is None or len(occupancy) < 200:
        print("\n⚠ Реальные данные не найдены, используем синтетические...")
        timestamps, occupancy = generate_synthetic_data(n_points=500)
        print(f"✓ Сгенерировано {len(occupancy)} записей")
    
    # Сравнение моделей
    comparator = ModelComparator(
        test_size=TEST_SIZE,
        lookback=LOOKBACK,
        forecast_horizon=FORECAST_HORIZON
    )
    
    results = comparator.compare(occupancy, timestamps)
    
    if results:
        # Генерация всех графиков
        visualizer = ResultsVisualizer(results)
        visualizer.plot_all()
        
        # Сохраняем результаты в CSV
        results_df = pd.DataFrame({
            'Метрика': list(results['RF_LSTM'].keys()),
            'RF+LSTM': list(results['RF_LSTM'].values()),
            'RF+SARIMA': list(results['RF_SARIMA'].values())
        })
        
        # Удаляем временные столбцы с predictions и actuals
        results_df = results_df[~results_df['Метрика'].isin(['predictions', 'actuals'])]
        
        csv_path = os.path.join(config.RESULTS_DIR, 'models_comparison_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\n📁 Результаты сохранены: {csv_path}")
        
        # Краткий вывод в консоль
        print("\n" + "="*70)
        print("КРАТКИЙ ИТОГ")
        print("="*70)
        print(f"🏆 Победитель: {results['winner']}")
        print(f"\nКлючевые метрики:")
        print(f"  RF+LSTM   - MAE: {results['RF_LSTM']['MAE']:.4f}, SMAPE: {results['RF_LSTM']['SMAPE (%)']:.2f}%, R²: {results['RF_LSTM']['R²']:.4f}")
        print(f"  RF+SARIMA - MAE: {results['RF_SARIMA']['MAE']:.4f}, SMAPE: {results['RF_SARIMA']['SMAPE (%)']:.2f}%, R²: {results['RF_SARIMA']['R²']:.4f}")
        
        print(f"\n📊 Детали по метрикам:")
        
        # MAE
        if results['RF_LSTM']['MAE'] < results['RF_SARIMA']['MAE']:
            print(f"  ✅ MAE лучше у RF+LSTM ({results['RF_LSTM']['MAE']:.4f} vs {results['RF_SARIMA']['MAE']:.4f})")
        else:
            print(f"  ✅ MAE лучше у RF+SARIMA ({results['RF_SARIMA']['MAE']:.4f} vs {results['RF_LSTM']['MAE']:.4f})")
        
        # SMAPE
        if results['RF_LSTM']['SMAPE (%)'] < results['RF_SARIMA']['SMAPE (%)']:
            print(f"  ✅ SMAPE лучше у RF+LSTM ({results['RF_LSTM']['SMAPE (%)']:.2f}% vs {results['RF_SARIMA']['SMAPE (%)']:.2f}%)")
        else:
            print(f"  ✅ SMAPE лучше у RF+SARIMA ({results['RF_SARIMA']['SMAPE (%)']:.2f}% vs {results['RF_LSTM']['SMAPE (%)']:.2f}%)")
        
        # R²
        if results['RF_LSTM']['R²'] > results['RF_SARIMA']['R²']:
            print(f"  ✅ R² лучше у RF+LSTM ({results['RF_LSTM']['R²']:.4f} vs {results['RF_SARIMA']['R²']:.4f})")
        else:
            print(f"  ✅ R² лучше у RF+SARIMA ({results['RF_SARIMA']['R²']:.4f} vs {results['RF_LSTM']['R²']:.4f})")
        
        # Время обучения
        if results['RF_LSTM']['train_time'] < results['RF_SARIMA']['train_time']:
            print(f"  ✅ Скорость обучения лучше у RF+LSTM ({results['RF_LSTM']['train_time']:.2f}с vs {results['RF_SARIMA']['train_time']:.2f}с)")
        else:
            print(f"  ✅ Скорость обучения лучше у RF+SARIMA ({results['RF_SARIMA']['train_time']:.2f}с vs {results['RF_LSTM']['train_time']:.2f}с)")
        
    else:
        print("\n❌ Ошибка при сравнении моделей")
    
    print("\n" + "="*70)
    print("РАБОТА ЗАВЕРШЕНА")
    print("="*70)


if __name__ == "__main__":
    main()