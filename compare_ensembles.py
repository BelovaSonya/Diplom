# compare_ensembles.py
"""
Сравнение комбинированных моделей прогнозирования:
1. Random Forest + LSTM
2. Random Forest + SARIMA

Метрики сравнения:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score
- Время обучения и предсказания
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
import seaborn as sns

import config

# Импорт моделей
import sys
sys.path.append(os.path.dirname(__file__))

from ensemble_rf_lstm_fixed import EnsembleRFLSTM, TimeSeriesDataLoader as LSTMLoader
from ensemble_rf_sarima_fixed import EnsembleRFSARIMA, TimeSeriesDataLoader as SARIMALoader


class ModelComparator:
    """Класс для сравнения моделей"""
    
    def __init__(self, test_size=0.2, lookback=24, forecast_horizon=12):
        """
        Args:
            test_size: доля тестовых данных
            lookback: окно истории
            forecast_horizon: горизонт прогноза
        """
        self.test_size = test_size
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.results = {}
        self.models = {}
        
    def prepare_data(self, time_series, timestamps=None):
        """Подготовка данных для тестирования"""
        if isinstance(time_series, pd.Series):
            time_series = time_series.values
        
        # Разделение на train/test
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
        
        # Создание модели
        model = EnsembleRFLSTM(
            lookback=self.lookback,
            forecast_horizon=self.forecast_horizon,
            model_dir=os.path.join(config.MODELS_DIR, "compare_rf_lstm")
        )
        
        # Обучение
        success = model.train(train_series, train_timestamps)
        
        train_time = time.time() - start_time
        
        if not success:
            print("❌ Ошибка обучения RF+LSTM")
            return None, None, None
        
        # Прогнозирование на тестовых данных
        print("\nПрогнозирование на тестовых данных...")
        predictions = []
        actuals = []
        
        start_pred_time = time.time()
        
        for i in range(len(test_series) - self.forecast_horizon):
            # Берем последние lookback значений для прогноза
            if i >= self.lookback:
                recent = test_series[i - self.lookback:i]
            else:
                recent = np.concatenate([train_series[-self.lookback + i:], test_series[:i]])
            
            current_time = test_timestamps[i] if test_timestamps is not None else None
            
            # Прогноз
            pred = model.predict(recent, current_time, return_all=False)
            
            if pred is not None:
                predictions.append(pred[0])
                actuals.append(test_series[i + 1])
        
        pred_time = time.time() - start_pred_time
        
        if len(predictions) == 0:
            print("❌ Не удалось сделать прогнозы")
            return None, None, None
        
        # Метрики
        metrics = self.calculate_metrics(actuals, predictions)
        metrics['train_time'] = train_time
        metrics['pred_time'] = pred_time
        
        print(f"\n✅ RF+LSTM результаты:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        return model, predictions, metrics
    
    def train_and_predict_rf_sarima(self, train_series, test_series, train_timestamps=None, test_timestamps=None):
        """Обучение и прогноз RF+SARIMA модели"""
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ МОДЕЛИ RF + SARIMA")
        print("="*60)
        
        start_time = time.time()
        
        # Создание модели
        model = EnsembleRFSARIMA(
            lookback=self.lookback,
            forecast_horizon=self.forecast_horizon,
            seasonal_period=24,
            model_dir=os.path.join(config.MODELS_DIR, "compare_rf_sarima"),
            ensemble_type='weighted'
        )
        
        # Обучение
        success = model.train(train_series, train_timestamps, optimize_sarima=False)
        
        train_time = time.time() - start_time
        
        if not success:
            print("❌ Ошибка обучения RF+SARIMA")
            return None, None, None
        
        # Прогнозирование на тестовых данных
        print("\nПрогнозирование на тестовых данных...")
        predictions = []
        actuals = []
        
        start_pred_time = time.time()
        
        for i in range(len(test_series) - self.forecast_horizon):
            # Берем последние lookback значений
            if i >= self.lookback:
                recent = test_series[i - self.lookback:i]
            else:
                recent = np.concatenate([train_series[-self.lookback + i:], test_series[:i]])
            
            current_time = test_timestamps[i] if test_timestamps is not None else None
            
            # Прогноз
            pred = model.predict(recent, current_time, return_all=False)
            
            if pred is not None:
                predictions.append(pred[0])
                actuals.append(test_series[i + 1])
        
        pred_time = time.time() - start_pred_time
        
        if len(predictions) == 0:
            print("❌ Не удалось сделать прогнозы")
            return None, None, None
        
        # Метрики
        metrics = self.calculate_metrics(actuals, predictions)
        metrics['train_time'] = train_time
        metrics['pred_time'] = pred_time
        
        print(f"\n✅ RF+SARIMA результаты:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        return model, predictions, metrics
    
    def calculate_metrics(self, y_true, y_pred):
        """Расчет метрик качества"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # MAE
        mae = mean_absolute_error(y_true, y_pred)
        
        # MSE
        mse = mean_squared_error(y_true, y_pred)
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # MAPE (избегаем деления на ноль)
        y_true_nonzero = y_true.copy()
        y_true_nonzero[y_true_nonzero == 0] = 0.01
        mape = np.mean(np.abs((y_true - y_pred) / y_true_nonzero)) * 100
        
        # R²
        r2 = r2_score(y_true, y_pred)
        
        # Дополнительные метрики
        absolute_errors = np.abs(y_true - y_pred)
        max_error = np.max(absolute_errors)
        
        # Смещение (bias)
        bias = np.mean(y_pred - y_true)
        
        # Точность в пределах 10% и 20%
        within_10pct = np.mean(absolute_errors / y_true_nonzero <= 0.1) * 100
        within_20pct = np.mean(absolute_errors / y_true_nonzero <= 0.2) * 100
        
        # SMAPE
        smape = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE (%)': mape,
            'SMAPE (%)': smape,
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
        
        # Подготовка данных
        train_series, test_series, train_ts, test_ts = self.prepare_data(time_series, timestamps)
        
        # Обучение и прогноз RF+LSTM
        model_lstm, preds_lstm, metrics_lstm = self.train_and_predict_rf_lstm(
            train_series, test_series, train_ts, test_ts
        )
        
        # Обучение и прогноз RF+SARIMA
        model_sarima, preds_sarima, metrics_sarima = self.train_and_predict_rf_sarima(
            train_series, test_series, train_ts, test_ts
        )
        
        # Определение победителя
        if metrics_lstm and metrics_sarima:
            winner = self.determine_winner(metrics_lstm, metrics_sarima)
            self.results = {
                'RF_LSTM': metrics_lstm,
                'RF_SARIMA': metrics_sarima,
                'winner': winner,
                'predictions': {
                    'actual': test_series[1:len(preds_lstm)+1],
                    'lstm': preds_lstm,
                    'sarima': preds_sarima
                }
            }
            return self.results
        
        return None
    
    def determine_winner(self, metrics_lstm, metrics_sarima):
        """Определение лучшей модели на основе метрик"""
        scores = {
            'RF+LSTM': 0,
            'RF+SARIMA': 0
        }
        
        # Сравнение по каждой метрике
        if metrics_lstm['MAE'] < metrics_sarima['MAE']:
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        if metrics_lstm['RMSE'] < metrics_sarima['RMSE']:
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        if metrics_lstm['MAPE (%)'] < metrics_sarima['MAPE (%)']:
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        if metrics_lstm['R²'] > metrics_sarima['R²']:
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        if abs(metrics_lstm['Bias']) < abs(metrics_sarima['Bias']):
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        if metrics_lstm['train_time'] < metrics_sarima['train_time']:
            scores['RF+LSTM'] += 1
        else:
            scores['RF+SARIMA'] += 1
        
        if scores['RF+LSTM'] > scores['RF+SARIMA']:
            return 'RF+LSTM'
        elif scores['RF+SARIMA'] > scores['RF+LSTM']:
            return 'RF+SARIMA'
        else:
            return 'Ничья'


class ComparisonVisualizer:
    """Визуализация результатов сравнения"""
    
    def __init__(self, results):
        self.results = results
        self.colors = {'lstm': '#2E86AB', 'sarima': '#D64933', 'actual': '#2C3E50'}
    
    def create_figure_1_metrics_table(self):
        """Первая картинка: сводная таблица метрик с выделением лучших значений"""
        
        if not self.results:
            print("Нет данных для визуализации")
            return None
        
        metrics_lstm = self.results['RF_LSTM']
        metrics_sarima = self.results['RF_SARIMA']
        
        # Создание фигуры для таблицы
        fig1, ax1 = plt.subplots(figsize=(14, 10))
        ax1.axis('tight')
        ax1.axis('off')
        
        # Выбор метрик для таблицы
        table_metrics = [
            'MAE', 'MSE', 'RMSE', 'MAPE (%)', 'SMAPE (%)', 
            'R²', 'Max Error', 'Bias', 'Within 10% (%)', 'Within 20% (%)',
            'train_time', 'pred_time'
        ]
        
        # Русские названия метрик
        metric_names_ru = {
            'MAE': 'MAE (средняя абс. ошибка)',
            'MSE': 'MSE (среднекв. ошибка)',
            'RMSE': 'RMSE (корень из MSE)',
            'MAPE (%)': 'MAPE (процент ошибки)',
            'SMAPE (%)': 'SMAPE (симм. MAPE)',
            'R²': 'R² (коэф. детерминации)',
            'Max Error': 'Максимальная ошибка',
            'Bias': 'Систематическое смещение',
            'Within 10% (%)': 'Точность в пределах 10%',
            'Within 20% (%)': 'Точность в пределах 20%',
            'train_time': 'Время обучения (сек)',
            'pred_time': 'Время прогноза (сек)'
        }
        
        # Формирование данных для таблицы
        cell_text = []
        
        for metric in table_metrics:
            val_lstm = metrics_lstm[metric]
            val_sarima = metrics_sarima[metric]
            metric_ru = metric_names_ru.get(metric, metric)
            
            # Определяем, какое значение лучше
            if metric in ['MAE', 'MSE', 'RMSE', 'MAPE (%)', 'SMAPE (%)', 'Max Error', 'train_time', 'pred_time']:
                lower_is_better = True
                best_is_lstm = val_lstm < val_sarima
            elif metric == 'R²':
                lower_is_better = False
                best_is_lstm = val_lstm > val_sarima
            elif metric == 'Bias':
                # Bias: ближе к нулю лучше
                best_is_lstm = abs(val_lstm) < abs(val_sarima)
                val_lstm_display = val_lstm
                val_sarima_display = val_sarima
            elif metric in ['Within 10% (%)', 'Within 20% (%)']:
                lower_is_better = False
                best_is_lstm = val_lstm > val_sarima
            else:
                lower_is_better = True
                best_is_lstm = val_lstm < val_sarima
            
            # Форматирование значений
            if metric in ['train_time', 'pred_time']:
                fmt_lstm = f"{val_lstm:.2f}"
                fmt_sarima = f"{val_sarima:.2f}"
            elif metric in ['MAPE (%)', 'SMAPE (%)', 'Within 10% (%)', 'Within 20% (%)']:
                fmt_lstm = f"{val_lstm:.2f}%"
                fmt_sarima = f"{val_sarima:.2f}%"
            elif metric == 'R²':
                fmt_lstm = f"{val_lstm:.4f}"
                fmt_sarima = f"{val_sarima:.4f}"
            elif metric == 'Bias':
                fmt_lstm = f"{val_lstm:.4f}"
                fmt_sarima = f"{val_sarima:.4f}"
            else:
                fmt_lstm = f"{val_lstm:.4f}"
                fmt_sarima = f"{val_sarima:.4f}"
            
            # Добавляем звездочку для лучшего показателя
            if best_is_lstm:
                fmt_lstm = f"★ {fmt_lstm}"
            else:
                fmt_sarima = f"★ {fmt_sarima}"
            
            cell_text.append([metric_ru, fmt_lstm, fmt_sarima])
        
        # Создание таблицы
        table = ax1.table(cellText=cell_text, 
                         colLabels=['Метрика', 'RF+LSTM', 'RF+SARIMA'],
                         cellLoc='center', 
                         loc='center',
                         colWidths=[0.4, 0.3, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        
        # Стилизация заголовков
        for i in range(3):
            table[(0, i)].set_facecolor('#34495E')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
        
        # Стилизация строк
        for i in range(1, len(cell_text) + 1):
            for j in range(3):
                if j == 0:
                    table[(i, j)].set_facecolor('#ECF0F1' if i % 2 == 0 else '#F8F9F9')
                    table[(i, j)].set_text_props(weight='bold', fontsize=10)
                else:
                    table[(i, j)].set_facecolor('#ECF0F1' if i % 2 == 0 else '#F8F9F9')
                    text = table[(i, j)].get_text().get_text()
                    if '★' in text:
                        table[(i, j)].set_text_props(weight='bold', color='#27AE60', fontsize=11)
                        table[(i, j)].get_text().set_text(text.replace('★ ', ''))
                    else:
                        table[(i, j)].set_text_props(color='#7F8C8D', fontsize=10)
        
        # Заголовок
        ax1.set_title(f'Сравнение метрик моделей прогнозирования\nПобедитель: {self.results["winner"]}', 
                     fontsize=16, fontweight='bold', pad=30)
        
        # Легенда
        legend_elements = [
            Patch(facecolor='#27AE60', alpha=0.3, label='★ Лучшее значение метрики')
        ]
        ax1.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
                  ncol=1, fontsize=11, frameon=True)
        
        plt.tight_layout()
        
        # Сохранение
        save_path = os.path.join(config.RESULTS_DIR, 'figure_1_metrics_table.png')
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\n📊 Рисунок 1 (таблица метрик) сохранен: {save_path}")
        
        return fig1
    
    def create_figure_2_graphs(self):
        """Вторая картинка: графики сравнения прогнозов и ошибок"""
        
        if not self.results:
            print("Нет данных для визуализации")
            return None
        
        metrics_lstm = self.results['RF_LSTM']
        metrics_sarima = self.results['RF_SARIMA']
        predictions = self.results.get('predictions', {})
        
        # Создание фигуры с сеткой 2x3
        fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        colors = {'lstm': '#2E86AB', 'sarima': '#D64933', 'actual': '#2C3E50'}
        
        # 1. График фактических значений и прогнозов
        ax1 = axes[0, 0]
        if predictions:
            actual = predictions['actual'][:150]
            lstm_pred = predictions['lstm'][:150]
            sarima_pred = predictions['sarima'][:150]
            
            ax1.plot(actual, 'o-', label='Фактические', color=colors['actual'], 
                    alpha=0.8, linewidth=1.5, markersize=3)
            ax1.plot(lstm_pred, 's--', label='RF+LSTM', color=colors['lstm'], 
                    alpha=0.8, linewidth=1.5, markersize=3)
            ax1.plot(sarima_pred, '^--', label='RF+SARIMA', color=colors['sarima'], 
                    alpha=0.8, linewidth=1.5, markersize=3)
            ax1.set_xlabel('Временной шаг', fontsize=11)
            ax1.set_ylabel('Загрузка (%)', fontsize=11)
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, alpha=0.3)
        
        # 2. График MAE ошибки по шагам
        ax2 = axes[0, 1]
        if predictions:
            actual_full = predictions['actual']
            lstm_pred_full = predictions['lstm']
            sarima_pred_full = predictions['sarima']
            
            errors_lstm = np.abs(np.array(actual_full) - np.array(lstm_pred_full))
            errors_sarima = np.abs(np.array(actual_full) - np.array(sarima_pred_full))
            
            steps = range(1, len(errors_lstm) + 1)
            ax2.plot(steps, errors_lstm, 'o-', label='RF+LSTM', color=colors['lstm'], 
                    alpha=0.7, linewidth=0.8, markersize=2)
            ax2.plot(steps, errors_sarima, 's-', label='RF+SARIMA', color=colors['sarima'], 
                    alpha=0.7, linewidth=0.8, markersize=2)
            
            # Средние линии
            mean_lstm = np.mean(errors_lstm)
            mean_sarima = np.mean(errors_sarima)
            ax2.axhline(y=mean_lstm, color=colors['lstm'], linestyle='--', 
                       alpha=0.7, label=f'Ср. RF+LSTM: {mean_lstm:.2f}')
            ax2.axhline(y=mean_sarima, color=colors['sarima'], linestyle='--', 
                       alpha=0.7, label=f'Ср. RF+SARIMA: {mean_sarima:.2f}')
            
            ax2.set_xlabel('Временной шаг', fontsize=11)
            ax2.set_ylabel('Абсолютная ошибка (%)', fontsize=11)
            ax2.legend(loc='upper left', fontsize=9)
            ax2.grid(True, alpha=0.3)
        
        # 3. Гистограмма распределения ошибок
        ax3 = axes[0, 2]
        if predictions:
            ax3.hist(errors_lstm, bins=30, alpha=0.6, label='RF+LSTM', 
                    color=colors['lstm'], edgecolor='black', linewidth=0.5)
            ax3.hist(errors_sarima, bins=30, alpha=0.6, label='RF+SARIMA', 
                    color=colors['sarima'], edgecolor='black', linewidth=0.5)
            ax3.set_xlabel('Абсолютная ошибка (%)', fontsize=11)
            ax3.set_ylabel('Частота', fontsize=11)
            ax3.legend(loc='upper right', fontsize=10)
            ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Сравнение метрик ошибок (барчарт)
        ax4 = axes[1, 0]
        compare_metrics = ['MAE', 'RMSE', 'MAPE (%)', 'SMAPE (%)']
        lstm_values = [metrics_lstm[m] for m in compare_metrics]
        sarima_values = [metrics_sarima[m] for m in compare_metrics]
        
        x = np.arange(len(compare_metrics))
        width = 0.35
        bars1 = ax4.bar(x - width/2, lstm_values, width, label='RF+LSTM', color=colors['lstm'])
        bars2 = ax4.bar(x + width/2, sarima_values, width, label='RF+SARIMA', color=colors['sarima'])
        
        # Добавление значений на столбцы
        for bar in bars1:
            height = bar.get_height()
            ax4.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax4.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        
        ax4.set_xlabel('Метрики ошибок', fontsize=11)
        ax4.set_ylabel('Значение', fontsize=11)
        ax4.set_xticks(x)
        ax4.set_xticklabels(compare_metrics)
        ax4.legend(loc='upper right', fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Точность прогнозов
        ax5 = axes[1, 1]
        accuracy_metrics = ['Within 10% (%)', 'Within 20% (%)']
        lstm_acc = [metrics_lstm.get('Within 10% (%)', 0), metrics_lstm.get('Within 20% (%)', 0)]
        sarima_acc = [metrics_sarima.get('Within 10% (%)', 0), metrics_sarima.get('Within 20% (%)', 0)]
        
        x = np.arange(len(accuracy_metrics))
        bars1 = ax5.bar(x - width/2, lstm_acc, width, label='RF+LSTM', color=colors['lstm'])
        bars2 = ax5.bar(x + width/2, sarima_acc, width, label='RF+SARIMA', color=colors['sarima'])
        
        for bar in bars1:
            height = bar.get_height()
            ax5.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax5.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        
        ax5.set_xlabel('Допуск ошибки', fontsize=11)
        ax5.set_ylabel('Процент прогнозов (%)', fontsize=11)
        ax5.set_xticks(x)
        ax5.set_xticklabels(['В пределах 10%', 'В пределах 20%'])
        ax5.legend(loc='lower right', fontsize=10)
        ax5.set_ylim(0, 105)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Сравнение R² и Bias
        ax6 = axes[1, 2]
        
        # Данные для grouped bar chart
        metrics_2 = ['R²', '|Bias|']
        lstm_vals = [metrics_lstm['R²'], abs(metrics_lstm['Bias'])]
        sarima_vals = [metrics_sarima['R²'], abs(metrics_sarima['Bias'])]
        
        x = np.arange(len(metrics_2))
        bars1 = ax6.bar(x - width/2, lstm_vals, width, label='RF+LSTM', color=colors['lstm'])
        bars2 = ax6.bar(x + width/2, sarima_vals, width, label='RF+SARIMA', color=colors['sarima'])
        
        for bar in bars1:
            height = bar.get_height()
            ax6.annotate(f'{height:.4f}' if height < 1 else f'{height:.2f}', 
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax6.annotate(f'{height:.4f}' if height < 1 else f'{height:.2f}', 
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        
        ax6.set_xlabel('Метрики', fontsize=11)
        ax6.set_ylabel('Значение', fontsize=11)
        ax6.set_xticks(x)
        ax6.set_xticklabels(['R² (выше = лучше)', '|Bias| (ниже = лучше)'])
        ax6.legend(loc='upper right', fontsize=10)
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Сохранение
        save_path = os.path.join(config.RESULTS_DIR, 'figure_2_graphs.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"📊 Рисунок 2 (графики) сохранен: {save_path}")
        
        return fig2
    
    def create_all_figures(self):
        """Создание всех фигур"""
        fig1 = self.create_figure_1_metrics_table()
        fig2 = self.create_figure_2_graphs()
        return fig1, fig2
    
    def print_summary(self):
        """Печать итогового отчета"""
        if not self.results:
            print("Нет данных для отчета")
            return
        
        print("\n" + "="*70)
        print("ИТОГОВЫЙ ОТЧЕТ СРАВНЕНИЯ")
        print("="*70)
        
        metrics_lstm = self.results['RF_LSTM']
        metrics_sarima = self.results['RF_SARIMA']
        winner = self.results['winner']
        
        print(f"\n📊 RF+LSTM МЕТРИКИ:")
        print(f"   MAE: {metrics_lstm['MAE']:.4f}")
        print(f"   RMSE: {metrics_lstm['RMSE']:.4f}")
        print(f"   MAPE: {metrics_lstm['MAPE (%)']:.2f}%")
        print(f"   R²: {metrics_lstm['R²']:.4f}")
        print(f"   Время обучения: {metrics_lstm['train_time']:.2f} сек")
        
        print(f"\n📊 RF+SARIMA МЕТРИКИ:")
        print(f"   MAE: {metrics_sarima['MAE']:.4f}")
        print(f"   RMSE: {metrics_sarima['RMSE']:.4f}")
        print(f"   MAPE: {metrics_sarima['MAPE (%)']:.2f}%")
        print(f"   R²: {metrics_sarima['R²']:.4f}")
        print(f"   Время обучения: {metrics_sarima['train_time']:.2f} сек")
        
        print(f"\n🏆 ПОБЕДИТЕЛЬ: {winner}")
        
        # Детальный анализ
        print("\n" + "="*70)
        print("ДЕТАЛЬНЫЙ АНАЛИЗ")
        print("="*70)
        
        if winner == "RF+LSTM":
            advantage_mae = ((metrics_sarima['MAE'] - metrics_lstm['MAE']) / metrics_sarima['MAE']) * 100
            advantage_rmse = ((metrics_sarima['RMSE'] - metrics_lstm['RMSE']) / metrics_sarima['RMSE']) * 100
            print(f"\n✅ RF+LSTM лучше на:")
            print(f"   MAE: на {advantage_mae:.1f}% меньше ошибка")
            print(f"   RMSE: на {advantage_rmse:.1f}% меньше ошибка")
        elif winner == "RF+SARIMA":
            advantage_mae = ((metrics_lstm['MAE'] - metrics_sarima['MAE']) / metrics_lstm['MAE']) * 100
            advantage_rmse = ((metrics_lstm['RMSE'] - metrics_sarima['RMSE']) / metrics_lstm['RMSE']) * 100
            print(f"\n✅ RF+SARIMA лучше на:")
            print(f"   MAE: на {advantage_mae:.1f}% меньше ошибка")
            print(f"   RMSE: на {advantage_rmse:.1f}% меньше ошибка")
        
        # Рекомендация
        print("\n" + "="*70)
        print("РЕКОМЕНДАЦИЯ")
        print("="*70)
        
        if winner == "RF+LSTM":
            print("\n🎯 Рекомендуется использовать модель RF+LSTM")
            print("   • Лучше для данных с нелинейными зависимостями")
            print("   • Хорошо捕捉 долгосрочные паттерны")
        elif winner == "RF+SARIMA":
            print("\n🎯 Рекомендуется использовать модель RF+SARIMA")
            print("   • Лучше для данных с явной сезонностью")
            print("   • Быстрее обучается")
            print("   • Интерпретируемые результаты")
        else:
            print("\n🎯 Обе модели показывают схожие результаты")
            print("   • Можно использовать ансамбль из обеих моделей")
            print("   • RF+LSTM лучше для сложных нелинейных зависимостей")
            print("   • RF+SARIMA лучше для сезонных данных и интерпретируемости")


def generate_synthetic_data(n_points=500, pattern='mixed'):
    """Генерация синтетических данных с разными паттернами"""
    np.random.seed(42)
    
    timestamps = [datetime(2024, 1, 1, 8, 0, 0) + timedelta(minutes=10 * i) for i in range(n_points)]
    
    if pattern == 'seasonal':
        hours = np.array([ts.hour + ts.minute/60 for ts in timestamps])
        occupancy = 30 + 45 * np.sin(2 * np.pi * hours / 24)
        occupancy += np.random.normal(0, 5, n_points)
    elif pattern == 'trend':
        occupancy = 20 + np.linspace(0, 60, n_points)
        occupancy += np.random.normal(0, 8, n_points)
    elif pattern == 'nonlinear':
        x = np.linspace(0, 4*np.pi, n_points)
        occupancy = 30 + 20 * np.sin(x) + 15 * np.sin(2*x) + 10 * np.cos(3*x)
        occupancy += np.random.normal(0, 6, n_points)
    else:
        hours = np.array([ts.hour + ts.minute/60 for ts in timestamps])
        seasonal = 30 + 35 * np.sin(2 * np.pi * hours / 24)
        trend = 5 * np.sin(np.linspace(0, np.pi, n_points))
        noise = np.random.normal(0, 5, n_points)
        occupancy = seasonal + trend + noise
    
    occupancy = np.clip(occupancy, 0, 100)
    
    return timestamps, occupancy


def main():
    """Главная функция сравнения"""
    
    print("\n" + "="*70)
    print("СРАВНЕНИЕ МОДЕЛЕЙ ПРОГНОЗИРОВАНИЯ")
    print("RF+LSTM vs RF+SARIMA")
    print("="*70)
    
    # Параметры
    LOOKBACK = 24
    FORECAST_HORIZON = 12
    TEST_SIZE = 0.2
    
    print(f"\nПараметры сравнения:")
    print(f"  Lookback: {LOOKBACK} шагов")
    print(f"  Горизонт прогноза: {FORECAST_HORIZON} шагов")
    print(f"  Тестовый размер: {TEST_SIZE * 100}%")
    
    # Загрузка реальных данных
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
    
    # Если нет реальных данных, используем синтетические
    if occupancy is None or len(occupancy) < 200:
        print("\n⚠ Реальные данные не найдены или недостаточны")
        print("Используем синтетические данные со смешанным паттерном...")
        timestamps, occupancy = generate_synthetic_data(n_points=500, pattern='mixed')
        print(f"✓ Сгенерировано {len(occupancy)} записей")
    
    # Сравнение моделей
    comparator = ModelComparator(
        test_size=TEST_SIZE,
        lookback=LOOKBACK,
        forecast_horizon=FORECAST_HORIZON
    )
    
    results = comparator.compare(occupancy, timestamps)
    
    if results:
        # Создание двух фигур
        visualizer = ComparisonVisualizer(results)
        fig1, fig2 = visualizer.create_all_figures()
        
        # Вывод отчета
        visualizer.print_summary()
        
        # Сохранение результатов в CSV
        results_df = pd.DataFrame({
            'Метрика': list(results['RF_LSTM'].keys()),
            'RF+LSTM': list(results['RF_LSTM'].values()),
            'RF+SARIMA': list(results['RF_SARIMA'].values())
        })
        
        csv_path = os.path.join(config.RESULTS_DIR, 'models_comparison_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"\n📁 Результаты сохранены: {csv_path}")
        
        print("\n" + "="*70)
        print("СОЗДАНЫ ДВА ИЗОБРАЖЕНИЯ:")
        print("  • figure_1_metrics_table.png - таблица метрик")
        print("  • figure_2_graphs.png - графики сравнения")
        print("="*70)
        
        plt.show()
        
    else:
        print("\n❌ Ошибка при сравнении моделей")
    
    print("\n" + "="*70)
    print("СРАВНЕНИЕ ЗАВЕРШЕНО")
    print("="*70)


if __name__ == "__main__":
    main()