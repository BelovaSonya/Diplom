# parking_visualizer.py
import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import joblib
from tensorflow.keras.models import load_model
import datetime
import random
from collections import deque
import threading
import time

# Конфигурация
class Config:
    IMG_SIZE = (224, 224)
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
    
    # Цвета для отображения
    COLORS = {
        'free': '#2ecc71',      # зеленый - свободно
        'occupied': '#e74c3c',   # красный - занято
        'background': '#ecf0f1', # светлый фон
        'text': '#2c3e50'        # темный текст
    }

class ParkingVisualizer:
    """Класс для визуализации парковки с прогнозированием"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.parking_spots = []  # список координат парковочных мест
        self.load_models()
        self.create_parking_layout()
        
    def load_models(self):
        """Загрузка сохраненных моделей"""
        print("Загрузка моделей...")
        
        # Загрузка Random Forest модели
        rf_path = os.path.join(Config.MODELS_DIR, "Random_Forest.pkl")
        if os.path.exists(rf_path):
            self.models['random_forest'] = joblib.load(rf_path)
            print("✓ Random Forest загружена")
        
        # Загрузка CNN модели
        cnn_path = os.path.join(Config.MODELS_DIR, "Simple_CNN.h5")
        if os.path.exists(cnn_path):
            self.models['cnn'] = load_model(cnn_path)
            print("✓ CNN модель загружена")
        
        # Загрузка скейлера
        scaler_path = os.path.join(Config.MODELS_DIR, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("✓ Скейлер загружен")
        
        if not self.models:
            print("⚠ Модели не найдены! Использую тестовые данные.")
    
    def create_parking_layout(self):
        """Создание макета парковки"""
        # Создаем сетку парковочных мест (20x2 для примера)
        rows = 10
        cols = 4
        
        spot_width = 80
        spot_height = 100
        start_x = 50
        start_y = 50
        spacing = 10
        
        for row in range(rows):
            for col in range(cols):
                x = start_x + col * (spot_width + spacing)
                y = start_y + row * (spot_height + spacing)
                self.parking_spots.append({
                    'id': row * cols + col,
                    'row': row,
                    'col': col,
                    'bbox': (x, y, spot_width, spot_height),
                    'occupied': random.choice([0, 1]),  # для теста, потом заменим
                    'prediction': None,
                    'free_time': None
                })
        
        print(f"Создано {len(self.parking_spots)} парковочных мест")
    
    def predict_spot(self, spot_image):
        """Предсказание занятости для одного места"""
        if not self.models:
            return random.random() > 0.5, random.random()
        
        try:
            # Подготовка изображения
            img = cv2.resize(spot_image, Config.IMG_SIZE)
            img = img.astype(np.float32) / 255.0
            
            # Извлечение признаков для Random Forest
            features = self.extract_features(img)
            
            predictions = []
            probabilities = []
            
            # Random Forest
            if 'random_forest' in self.models and self.scaler:
                features_scaled = self.scaler.transform([features])
                pred_rf = self.models['random_forest'].predict(features_scaled)[0]
                prob_rf = self.models['random_forest'].predict_proba(features_scaled)[0][1]
                predictions.append(pred_rf)
                probabilities.append(prob_rf)
            
            # CNN
            if 'cnn' in self.models:
                img_batch = np.expand_dims(img, axis=0)
                prob_cnn = float(self.models['cnn'].predict(img_batch, verbose=0)[0][0])
                pred_cnn = 1 if prob_cnn > 0.5 else 0
                predictions.append(pred_cnn)
                probabilities.append(prob_cnn)
            
            # Ансамбль
            if predictions:
                final_pred = int(np.mean(predictions) > 0.5)
                final_prob = np.mean(probabilities)
                return final_pred, final_prob
            
            return random.random() > 0.5, random.random()
            
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return random.random() > 0.5, random.random()
    
    def extract_features(self, img):
        """Извлечение признаков из изображения"""
        mean_rgb = np.mean(img, axis=(0, 1))
        std_rgb = np.std(img, axis=(0, 1))
        
        hist_features = []
        for i in range(3):
            hist = np.histogram(img[:,:,i], bins=10, range=(0, 1))[0]
            hist_features.extend(hist)
        
        gray = np.mean(img, axis=2)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        return np.concatenate([
            mean_rgb, std_rgb, 
            hist_features,
            [np.mean(gradient_magnitude), np.std(gradient_magnitude)]
        ])
    
    def predict_parking_status(self):
        """Прогнозирование статуса всех парковочных мест"""
        occupied_count = 0
        free_count = 0
        
        for spot in self.parking_spots:
            # Здесь должна быть загрузка реального изображения места
            # Для демо используем случайные значения
            if random.random() > 0.7:
                spot['occupied'] = 1
                occupied_count += 1
                # Прогноз освобождения (в минутах)
                spot['free_time'] = random.randint(5, 60)
            else:
                spot['occupied'] = 0
                free_count += 1
                spot['free_time'] = None
        
        return occupied_count, free_count
    
    def predict_when_free(self, occupied_spots):
        """Прогноз когда освободятся места"""
        if occupied_spots == 0:
            return "Все места свободны! 🎉"
        
        # Используем простую модель прогнозирования
        # В реальности нужно использовать временные ряды
        
        # Предполагаем, что освобождение происходит по экспоненциальному распределению
        avg_free_time = 15  # среднее время освобождения в минутах
        
        # Прогнозируем, когда освободится первое место
        first_free = random.randint(2, 10)
        
        if occupied_spots == len(self.parking_spots):
            return f"⚠ Все места заняты! Первое место освободится через {first_free} минут."
        else:
            free_rate = 1 - (occupied_spots / len(self.parking_spots))
            if free_rate < 0.3:
                return f"🟡 Осталось {occupied_spots} мест. Первое место освободится через ~{first_free} минут."
            else:
                return f"🟢 Доступно {len(self.parking_spots) - occupied_spots} мест. Места появятся через {first_free//2}-{first_free} минут."

class ParkingApp:
    """Главное приложение с GUI"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Система прогнозирования парковки")
        self.root.geometry("1200x700")
        self.root.configure(bg=Config.COLORS['background'])
        
        self.visualizer = ParkingVisualizer()
        self.setup_ui()
        
        # Автоматическое обновление каждые 10 секунд
        self.update_timer = None
        self.start_auto_update()
        
    def setup_ui(self):
        """Настройка интерфейса"""
        # Заголовок
        title_frame = tk.Frame(self.root, bg=Config.COLORS['background'])
        title_frame.pack(fill=tk.X, padx=20, pady=10)
        
        title_label = tk.Label(
            title_frame,
            text="🚗 Система мониторинга и прогнозирования парковки 🚙",
            font=('Arial', 20, 'bold'),
            bg=Config.COLORS['background'],
            fg=Config.COLORS['text']
        )
        title_label.pack()
        
        # Основной фрейм для canvas и информации
        main_frame = tk.Frame(self.root, bg=Config.COLORS['background'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Canvas для отрисовки парковки
        self.canvas = tk.Canvas(
            main_frame,
            width=900,
            height=600,
            bg='white',
            highlightthickness=2,
            highlightbackground='gray'
        )
        self.canvas.pack(side=tk.LEFT, padx=10)
        
        # Панель информации справа
        info_frame = tk.Frame(main_frame, bg=Config.COLORS['background'], width=300)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)
        
        # Статистика
        stats_frame = tk.LabelFrame(
            info_frame,
            text="📊 Статистика",
            font=('Arial', 12, 'bold'),
            bg=Config.COLORS['background'],
            fg=Config.COLORS['text']
        )
        stats_frame.pack(fill=tk.X, pady=10)
        
        self.total_label = tk.Label(
            stats_frame,
            text="Всего мест: 0",
            font=('Arial', 11),
            bg=Config.COLORS['background']
        )
        self.total_label.pack(anchor=tk.W, padx=10, pady=5)
        
        self.occupied_label = tk.Label(
            stats_frame,
            text="Занято: 0",
            font=('Arial', 11),
            bg=Config.COLORS['background'],
            fg=Config.COLORS['occupied']
        )
        self.occupied_label.pack(anchor=tk.W, padx=10, pady=5)
        
        self.free_label = tk.Label(
            stats_frame,
            text="Свободно: 0",
            font=('Arial', 11),
            bg=Config.COLORS['background'],
            fg=Config.COLORS['free']
        )
        self.free_label.pack(anchor=tk.W, padx=10, pady=5)
        
        self.occupancy_rate = tk.Label(
            stats_frame,
            text="Загрузка: 0%",
            font=('Arial', 11),
            bg=Config.COLORS['background']
        )
        self.occupancy_rate.pack(anchor=tk.W, padx=10, pady=5)
        
        # Прогноз
        prediction_frame = tk.LabelFrame(
            info_frame,
            text="🔮 Прогноз",
            font=('Arial', 12, 'bold'),
            bg=Config.COLORS['background'],
            fg=Config.COLORS['text']
        )
        prediction_frame.pack(fill=tk.X, pady=10)
        
        self.prediction_label = tk.Label(
            prediction_frame,
            text="Обновление...",
            font=('Arial', 11),
            bg=Config.COLORS['background'],
            wraplength=280,
            justify=tk.LEFT
        )
        self.prediction_label.pack(padx=10, pady=10)
        
        # Кнопки управления
        control_frame = tk.Frame(info_frame, bg=Config.COLORS['background'])
        control_frame.pack(fill=tk.X, pady=10)
        
        update_btn = tk.Button(
            control_frame,
            text="🔄 Обновить",
            command=self.update_parking_status,
            font=('Arial', 10, 'bold'),
            bg='#3498db',
            fg='white',
            padx=20,
            pady=5
        )
        update_btn.pack(fill=tk.X, pady=5)
        
        auto_update_var = tk.BooleanVar(value=True)
        auto_update_cb = tk.Checkbutton(
            control_frame,
            text="Автообновление (10 сек)",
            variable=auto_update_var,
            command=lambda: self.toggle_auto_update(auto_update_var.get()),
            bg=Config.COLORS['background']
        )
        auto_update_cb.pack(pady=5)
        
        # Статус
        self.status_label = tk.Label(
            info_frame,
            text="✅ Система активна",
            font=('Arial', 9),
            bg=Config.COLORS['background'],
            fg='green'
        )
        self.status_label.pack(pady=10)
        
        # Первое обновление
        self.update_parking_status()
    
    def draw_parking_lot(self):
        """Отрисовка парковки на canvas"""
        self.canvas.delete("all")
        
        # Рисуем парковочные места
        for spot in self.visualizer.parking_spots:
            x, y, w, h = spot['bbox']
            
            # Выбираем цвет в зависимости от статуса
            if spot['occupied']:
                color = Config.COLORS['occupied']
                status_text = "ЗАНЯТО"
            else:
                color = Config.COLORS['free']
                status_text = "СВОБОДНО"
            
            # Рисуем прямоугольник места
            self.canvas.create_rectangle(
                x, y, x + w, y + h,
                fill=color,
                outline='black',
                width=2
            )
            
            # Номер места
            self.canvas.create_text(
                x + w//2, y + h//2 - 10,
                text=f"{spot['id']}",
                font=('Arial', 8, 'bold'),
                fill='white'
            )
            
            # Статус
            self.canvas.create_text(
                x + w//2, y + h//2 + 10,
                text=status_text,
                font=('Arial', 8),
                fill='white'
            )
            
            # Если занято, показываем прогноз освобождения
            if spot['occupied'] and spot.get('free_time'):
                self.canvas.create_text(
                    x + w//2, y + h - 10,
                    text=f"~{spot['free_time']} мин",
                    font=('Arial', 7),
                    fill='white'
                )
    
    def update_parking_status(self):
        """Обновление статуса парковки"""
        try:
            # Получаем статус парковки
            occupied, free = self.visualizer.predict_parking_status()
            total = occupied + free
            
            # Обновляем статистику
            self.total_label.config(text=f"Всего мест: {total}")
            self.occupied_label.config(text=f"Занято: {occupied}")
            self.free_label.config(text=f"Свободно: {free}")
            
            rate = (occupied / total * 100) if total > 0 else 0
            self.occupancy_rate.config(text=f"Загрузка: {rate:.1f}%")
            
            # Получаем прогноз
            prediction = self.visualizer.predict_when_free(occupied)
            self.prediction_label.config(text=prediction)
            
            # Отрисовываем парковку
            self.draw_parking_lot()
            
            # Обновляем статус
            self.status_label.config(
                text=f"✅ Обновлено: {datetime.datetime.now().strftime('%H:%M:%S')}",
                fg='green'
            )
            
        except Exception as e:
            self.status_label.config(text=f"❌ Ошибка: {str(e)}", fg='red')
    
    def toggle_auto_update(self, enabled):
        """Включение/отключение автообновления"""
        if enabled:
            self.start_auto_update()
        else:
            if self.update_timer:
                self.root.after_cancel(self.update_timer)
                self.update_timer = None
    
    def start_auto_update(self):
        """Запуск автообновления"""
        self.update_parking_status()
        self.update_timer = self.root.after(10000, self.start_auto_update)
    
    def run(self):
        """Запуск приложения"""
        self.root.mainloop()

# Запуск приложения
if __name__ == "__main__":
    app = ParkingApp()
    app.run()