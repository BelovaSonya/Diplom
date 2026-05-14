# parking_visualizer_advanced.py
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from ttkbootstrap.toast import ToastNotification
import joblib
from tensorflow.keras.models import load_model
import datetime
import random
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Конфигурация цветов
class Colors:
    PRIMARY = "#4CAF50"
    SECONDARY = "#81C784"
    BG_LIGHT = "#FFFFFF"
    BG_SECONDARY = "#F5F5F5"
    SUCCESS = "#2ecc71"
    DANGER = "#e74c3c"
    WARNING = "#f39c12"
    TEXT = "#2c3e50"
    TEXT_LIGHT = "#7f8c8d"
    ACCENT = "#3498db"
    DISABLED = "#95a5a6"
    
class ParkingSpot:
    """Класс парковочного места"""
    def __init__(self, spot_id, x, y, width, height, spot_type="normal"):
        self.id = spot_id
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.spot_type = spot_type
        self.occupied = False
        self.probability = 0.0
        self.free_time = None

class AdvancedParkingVisualizer:
    """Продвинутый визуализатор парковки"""
    
    def __init__(self):
        self.root = tb.Window(themename="litera")
        self.root.title("🚗 Парковки Москвы | Система мониторинга")
        self.root.geometry("1700x950")  # Увеличил размер окна
        self.root.minsize(1500, 850)
        
        self.setup_styles()
        self.models = {}
        self.load_models()
        
        # Создание парковочных мест
        self.parking_spots = []
        self.create_parking_layout()
        
        # Для уведомлений
        self.last_notification = ""
        self.notification_count = 0
        
        # Настройка интерфейса
        self.setup_ui()
        
        # Автообновление
        self.start_auto_update()
        
        # Запуск уведомлений
        self.start_notifications()
        
    def setup_styles(self):
        style = ttk.Style()
        style.configure('Header.TLabel', font=('Segoe UI', 28, 'bold'))
        style.configure('Stat.TLabel', font=('Segoe UI', 12))
        style.configure('BigStat.TLabel', font=('Segoe UI', 32, 'bold'))
        
    def load_models(self):
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        try:
            rf_path = os.path.join(models_dir, "Random_Forest.pkl")
            if os.path.exists(rf_path):
                self.models['random_forest'] = joblib.load(rf_path)
        except Exception as e:
            print(f"Ошибка загрузки моделей: {e}")
            
    def create_parking_layout(self):
        """Создание компактной парковки 15 мест + 3 для инвалидов"""
        spot_width = 95
        spot_height = 95
        spacing = 15
        
        start_x = 280
        start_y = 100
        
        spot_id = 0
        
        # Ряд 1: 5 обычных мест
        for i in range(5):
            x = start_x + i * (spot_width + spacing)
            y = start_y
            self.parking_spots.append(ParkingSpot(spot_id, x, y, spot_width, spot_height, "normal"))
            spot_id += 1
        
        # Ряд 2: 5 обычных мест
        for i in range(5):
            x = start_x + i * (spot_width + spacing)
            y = start_y + spot_height + spacing + 8
            self.parking_spots.append(ParkingSpot(spot_id, x, y, spot_width, spot_height, "normal"))
            spot_id += 1
        
        # Ряд 3: 5 обычных мест
        for i in range(5):
            x = start_x + i * (spot_width + spacing)
            y = start_y + 2 * (spot_height + spacing) + 16
            self.parking_spots.append(ParkingSpot(spot_id, x, y, spot_width, spot_height, "normal"))
            spot_id += 1
        
        # Места для инвалидов (слева)
        disabled_start_x = 120
        disabled_start_y = 150
        
        for i in range(3):
            x = disabled_start_x
            y = disabled_start_y + i * (spot_height + spacing)
            self.parking_spots.append(ParkingSpot(spot_id, x, y, spot_width, spot_height, "disabled"))
            spot_id += 1
        
        print(f"Создано {len(self.parking_spots)} мест (3 для инвалидов)")
        
    def setup_ui(self):
        """Настройка интерфейса - СЛЕВА СТАТИСТИКА, ЦЕНТР КАРТА, СПРАВА РЕКОМЕНДАЦИИ"""
        main_container = tb.Frame(self.root)
        main_container.pack(fill=BOTH, expand=True, padx=20, pady=20)
        
        # Верхняя панель
        header_frame = tb.Frame(main_container, bootstyle="light")
        header_frame.pack(fill=X, pady=(0, 20))
        
        # Центрированный заголовок
        header_center = tb.Frame(header_frame)
        header_center.pack(expand=True)
        
        title_label = tb.Label(
            header_center,
            text="🚗 ПАРКОВКИ МОСКВЫ",
            font=('Segoe UI', 28, 'bold'),
            foreground=Colors.PRIMARY
        )
        title_label.pack()
        
        subtitle_label = tb.Label(
            header_center,
            text="Система интеллектуального мониторинга парковочных мест",
            font=('Segoe UI', 10),
            foreground=Colors.TEXT_LIGHT
        )
        subtitle_label.pack()
        
        # Время справа
        time_frame = tb.Frame(header_frame)
        time_frame.pack(side=RIGHT, anchor='ne')
        
        self.clock_label = tb.Label(time_frame, font=('Segoe UI', 14, 'bold'), foreground=Colors.PRIMARY)
        self.clock_label.pack()
        self.date_label = tb.Label(time_frame, font=('Segoe UI', 10), foreground=Colors.TEXT_LIGHT)
        self.date_label.pack()
        self.update_clock()
        
        # Основной контент - 3 колонки
        content_frame = tb.Frame(main_container)
        content_frame.pack(fill=BOTH, expand=True)
        
        # ========== ЛЕВАЯ КОЛОНКА (Статистика + Прогноз) - УВЕЛИЧЕНА ==========
        left_panel = tb.Frame(content_frame, bootstyle="light", width=420)
        left_panel.pack(side=LEFT, fill=Y, padx=(0, 15))
        left_panel.pack_propagate(False)
        
        # Блок статистики (уменьшен)
        stats_card = tb.Frame(left_panel, bootstyle="light", relief="ridge", borderwidth=1)
        stats_card.pack(fill=X, pady=(0, 12))
        
        tb.Label(stats_card, text="📊 СТАТИСТИКА", font=('Segoe UI', 13, 'bold'), 
                foreground=Colors.PRIMARY).pack(pady=(12, 8))
        
        # Прогресс-бар загрузки
        occupancy_frame = tb.Frame(stats_card)
        occupancy_frame.pack(fill=X, padx=20, pady=8)
        
        tb.Label(occupancy_frame, text="Загрузка парковки", font=('Segoe UI', 10)).pack()
        self.occupancy_bar = ttk.Progressbar(occupancy_frame, length=340, mode='determinate', bootstyle="success-striped")
        self.occupancy_bar.pack(pady=5)
        self.occupancy_percent = tb.Label(occupancy_frame, font=('Segoe UI', 24, 'bold'), foreground=Colors.PRIMARY)
        self.occupancy_percent.pack()
        
        # Три основных показателя
        metrics_frame = tb.Frame(stats_card)
        metrics_frame.pack(fill=X, padx=20, pady=10)
        
        total_frame = tb.Frame(metrics_frame)
        total_frame.pack(side=LEFT, expand=True)
        tb.Label(total_frame, text="Всего", font=('Segoe UI', 9)).pack()
        self.total_label = tb.Label(total_frame, text="0", font=('Segoe UI', 22, 'bold'))
        self.total_label.pack()
        
        occupied_frame = tb.Frame(metrics_frame)
        occupied_frame.pack(side=LEFT, expand=True)
        tb.Label(occupied_frame, text="Занято", font=('Segoe UI', 9)).pack()
        self.occupied_label = tb.Label(occupied_frame, text="0", font=('Segoe UI', 22, 'bold'), foreground=Colors.DANGER)
        self.occupied_label.pack()
        
        free_frame = tb.Frame(metrics_frame)
        free_frame.pack(side=LEFT, expand=True)
        tb.Label(free_frame, text="Свободно", font=('Segoe UI', 9)).pack()
        self.free_label = tb.Label(free_frame, text="0", font=('Segoe UI', 22, 'bold'), foreground=Colors.SUCCESS)
        self.free_label.pack()
        
        # Места для инвалидов
        disabled_frame = tb.Frame(stats_card)
        disabled_frame.pack(fill=X, padx=20, pady=8)
        tb.Label(disabled_frame, text="♿ Места для инвалидов", font=('Segoe UI', 10)).pack()
        self.disabled_label = tb.Label(disabled_frame, text="0 свободно", font=('Segoe UI', 14, 'bold'), foreground=Colors.ACCENT)
        self.disabled_label.pack()
        
        # Блок прогнозирования (увеличен, чтобы было видно полностью)
        self.create_prediction_block(left_panel)
        
        # График динамики (уменьшен)
        self.setup_graph(left_panel)
        
        # ========== ЦЕНТРАЛЬНАЯ КОЛОНКА (Карта парковки) ==========
        center_panel = tb.Frame(content_frame, bootstyle="light")
        center_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 15))
        
        # Канвас для карты
        self.canvas = tk.Canvas(
            center_panel,
            bg=Colors.BG_LIGHT,
            highlightthickness=2,
            highlightbackground='#ddd',
            width=750,
            height=620
        )
        self.canvas.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # ========== ПРАВАЯ КОЛОНКА (Рекомендации + Уведомления) ==========
        right_panel = tb.Frame(content_frame, bootstyle="light", width=380)
        right_panel.pack(side=RIGHT, fill=Y)
        right_panel.pack_propagate(False)
        
        # Блок рекомендаций
        self.create_recommendations_block(right_panel)
        
        # Блок уведомлений
        self.create_notifications_block(right_panel)
        
        # ========== НИЖНЯЯ ПАНЕЛЬ (Кнопки) ==========
        bottom_frame = tb.Frame(main_container)
        bottom_frame.pack(fill=X, pady=(20, 0))
        
        btn_frame = tb.Frame(bottom_frame)
        btn_frame.pack()
        
        self.update_btn = tb.Button(btn_frame, text="🔄 Обновить", command=self.update_parking_status,
                                    bootstyle="primary", width=15)
        self.update_btn.pack(side=LEFT, padx=5)
        
        self.auto_update_var = tk.BooleanVar(value=True)
        self.auto_btn = tb.Button(btn_frame, text="⏸ Пауза", command=self.toggle_auto_update,
                                   bootstyle="warning", width=15)
        self.auto_btn.pack(side=LEFT, padx=5)
        
        self.reset_btn = tb.Button(btn_frame, text="🗑 Сброс", command=self.reset_parking,
                                    bootstyle="secondary", width=15)
        self.reset_btn.pack(side=LEFT, padx=5)
        
        self.notify_btn = tb.Button(btn_frame, text="🔔 Тест уведомления", command=self.test_notification,
                                     bootstyle="info", width=18)
        self.notify_btn.pack(side=LEFT, padx=5)
        
        # Статус
        self.status_bar = tb.Label(main_container, text="✅ Система активна | Уведомления каждые 30 сек", 
                                    font=('Segoe UI', 9), bootstyle="light")
        self.status_bar.pack(fill=X, pady=(10, 0))
        
    def create_recommendations_block(self, parent):
        """Блок рекомендаций (справа)"""
        tips_card = tb.Frame(parent, bootstyle="light", relief="ridge", borderwidth=1)
        tips_card.pack(fill=X, pady=(0, 12))
        
        header_frame = tb.Frame(tips_card)
        header_frame.pack(fill=X, pady=(12, 8))
        
        tb.Label(header_frame, text="💡", font=('Segoe UI', 20)).pack(side=LEFT, padx=(12, 5))
        tb.Label(header_frame, text="РЕКОМЕНДАЦИИ", font=('Segoe UI', 13, 'bold'),
                foreground=Colors.PRIMARY).pack(side=LEFT)
        
        self.tips_label = tb.Label(
            tips_card, 
            text="Загрузка данных...", 
            font=('Segoe UI', 10),
            wraplength=330, 
            justify=LEFT,
            foreground=Colors.TEXT
        )
        self.tips_label.pack(padx=15, pady=12)
        
    def create_notifications_block(self, parent):
        """Блок уведомлений (справа)"""
        notify_card = tb.Frame(parent, bootstyle="light", relief="ridge", borderwidth=1)
        notify_card.pack(fill=X)
        
        header_frame = tb.Frame(notify_card)
        header_frame.pack(fill=X, pady=(12, 8))
        
        tb.Label(header_frame, text="🔔", font=('Segoe UI', 18)).pack(side=LEFT, padx=(12, 5))
        tb.Label(header_frame, text="УВЕДОМЛЕНИЯ", font=('Segoe UI', 12, 'bold'),
                foreground=Colors.ACCENT).pack(side=LEFT)
        
        last_frame = tb.Frame(notify_card)
        last_frame.pack(fill=X, padx=15, pady=8)
        
        tb.Label(last_frame, text="Последнее уведомление:", font=('Segoe UI', 9)).pack()
        self.last_notification_label = tb.Label(
            last_frame, 
            text="--", 
            font=('Segoe UI', 9),
            wraplength=320,
            justify=LEFT,
            foreground=Colors.TEXT_LIGHT
        )
        self.last_notification_label.pack(pady=5)
        
        self.notify_count_label = tb.Label(
            notify_card,
            text="Уведомлений получено: 0",
            font=('Segoe UI', 9),
            foreground=Colors.TEXT_LIGHT
        )
        self.notify_count_label.pack(pady=(0, 12))
        
    def create_prediction_block(self, parent):
        """Блок прогнозирования (слева) - УВЕЛИЧЕН ДЛЯ ПОЛНОГО ОТОБРАЖЕНИЯ"""
        pred_card = tb.Frame(parent, bootstyle="light", relief="ridge", borderwidth=1)
        pred_card.pack(fill=X, pady=(0, 12))
        
        # Заголовок
        header_frame = tb.Frame(pred_card)
        header_frame.pack(fill=X, pady=(12, 8))
        
        tb.Label(header_frame, text="⏰", font=('Segoe UI', 20)).pack(side=LEFT, padx=(12, 5))
        tb.Label(header_frame, text="ПРОГНОЗ ОСВОБОЖДЕНИЯ", font=('Segoe UI', 12, 'bold'),
                foreground=Colors.ACCENT).pack(side=LEFT)
        
        # Время до полного освобождения
        full_frame = tb.Frame(pred_card)
        full_frame.pack(fill=X, padx=15, pady=8)
        
        tb.Label(full_frame, text="До полного освобождения:", font=('Segoe UI', 10)).pack()
        self.full_free_time = tb.Label(full_frame, text="--", font=('Segoe UI', 28, 'bold'),
                                        foreground=Colors.PRIMARY)
        self.full_free_time.pack()
        
        # Ближайшее освобождение
        next_frame = tb.Frame(pred_card)
        next_frame.pack(fill=X, padx=15, pady=8)
        
        tb.Label(next_frame, text="Ближайшее освобождение:", font=('Segoe UI', 10)).pack()
        self.next_free_label = tb.Label(next_frame, text="-- минут", font=('Segoe UI', 22, 'bold'),
                                         foreground=Colors.WARNING)
        self.next_free_label.pack()
        
        # Прогноз по часам
        hourly_frame = tb.Frame(pred_card)
        hourly_frame.pack(fill=X, padx=15, pady=8)
        
        tb.Label(hourly_frame, text="Прогноз загрузки по часам:", font=('Segoe UI', 10)).pack()
        
        hours_frame = tb.Frame(hourly_frame)
        hours_frame.pack(pady=6)
        
        self.hourly_labels = []
        hours = ["+1ч", "+2ч", "+3ч", "+4ч"]
        for i, hour in enumerate(hours):
            h_frame = tb.Frame(hours_frame)
            h_frame.pack(side=LEFT, padx=10)
            tb.Label(h_frame, text=hour, font=('Segoe UI', 9, 'bold')).pack()
            label = tb.Label(h_frame, text="--%", font=('Segoe UI', 11), foreground=Colors.TEXT_LIGHT)
            label.pack()
            self.hourly_labels.append(label)
        
        # Сообщение
        self.prediction_message = tb.Label(pred_card, text="", font=('Segoe UI', 10), 
                                            wraplength=360, justify=CENTER, foreground=Colors.TEXT)
        self.prediction_message.pack(pady=8)
        
    def setup_graph(self, parent):
        """Настройка графика - УМЕНЬШЕН"""
        graph_card = tb.Frame(parent, bootstyle="light", relief="ridge", borderwidth=1)
        graph_card.pack(fill=X)
        
        tb.Label(graph_card, text="📈 ДИНАМИКА ЗАГРУЗКИ", font=('Segoe UI', 12, 'bold'),
                foreground=Colors.PRIMARY).pack(pady=(10, 5))
        
        self.fig = Figure(figsize=(4.8, 1.8), dpi=80, facecolor='white')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#f8f9fa')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        
        self.occupancy_history = deque(maxlen=15)
        self.line, = self.ax.plot([], [], color=Colors.PRIMARY, linewidth=2)
        
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=graph_card)
        self.canvas_graph.draw()
        self.canvas_graph.get_tk_widget().pack(padx=12, pady=8, fill=BOTH, expand=True)
        
    def update_clock(self):
        now = datetime.now()
        self.clock_label.config(text=now.strftime("%H:%M:%S"))
        self.date_label.config(text=now.strftime("%d %B %Y"))
        self.root.after(1000, self.update_clock)
        
    def toggle_auto_update(self):
        if self.auto_update_var.get():
            self.auto_update_var.set(False)
            self.auto_btn.config(text="▶ Старт", bootstyle="success")
            self.status_bar.config(text="⏸ Автообновление остановлено")
        else:
            self.auto_update_var.set(True)
            self.auto_btn.config(text="⏸ Пауза", bootstyle="warning")
            self.start_auto_update()
            self.status_bar.config(text="✅ Система активна | Автообновление запущено")
            
    def start_auto_update(self):
        if self.auto_update_var.get():
            self.update_parking_status()
            self.root.after(10000, self.start_auto_update)
    
    def start_notifications(self):
        """Запуск периодических уведомлений каждые 30 секунд"""
        self.send_notification()
        self.root.after(30000, self.start_notifications)
    
    def send_notification(self):
        """Отправка всплывающего уведомления"""
        try:
            total = len(self.parking_spots)
            occupied = sum(1 for spot in self.parking_spots if spot.occupied)
            occupancy_rate = (occupied / total * 100) if total > 0 else 0
            disabled_free = sum(1 for spot in self.parking_spots 
                               if spot.spot_type == "disabled" and not spot.occupied)
            
            message = self.get_notification_message(occupancy_rate, occupied, disabled_free)
            
            # Показываем уведомление
            toast = ToastNotification(
                title="🚗 Парковки Москвы",
                message=message,
                duration=5000,
                bootstyle="info"
            )
            toast.show_toast()
            
            # Обновляем блок уведомлений
            self.last_notification_label.config(text=message)
            self.notification_count += 1
            self.notify_count_label.config(text=f"Уведомлений получено: {self.notification_count}")
            
            self.status_bar.config(
                text=f"✅ Уведомление отправлено | Всего: {self.notification_count}"
            )
            
        except Exception as e:
            print(f"Ошибка уведомления: {e}")
    
    def get_notification_message(self, occupancy_rate, occupied, disabled_free):
        """Формирование текста уведомления"""
        total = len(self.parking_spots)
        free = total - occupied
        
        if occupancy_rate < 30:
            return f"✅ Отлично! Свободно {free} мест. Можете спокойно парковаться!"
        elif occupancy_rate < 70:
            return f"🟡 Загрузка {occupancy_rate:.0f}%. Свободно {free} мест. Рекомендуем центральный ряд."
        elif occupancy_rate < 90:
            return f"🔴 Загрузка {occupancy_rate:.0f}%. Осталось {free} мест. Ожидайте освобождения через 10-15 мин."
        else:
            next_free = self.predict_next_free(occupied)[0]
            return f"⚠️ Парковка почти полная! Освобождение через {next_free} мин. ♿ Мест для инвалидов: {disabled_free}"
    
    def test_notification(self):
        """Тестовое уведомление"""
        toast = ToastNotification(
            title="🔔 Тестовое уведомление",
            message="Система уведомлений работает корректно!",
            duration=3000,
            bootstyle="success"
        )
        toast.show_toast()
        
    def predict_full_free_time(self, occupancy_rate):
        if occupancy_rate == 0:
            return "Уже свободна"
        elif occupancy_rate > 90:
            return "30-45 мин"
        elif occupancy_rate > 70:
            return "20-30 мин"
        elif occupancy_rate > 50:
            return "15-20 мин"
        elif occupancy_rate > 30:
            return "10-15 мин"
        else:
            return "5-10 мин"
    
    def predict_next_free(self, occupied_count):
        if occupied_count == 0:
            return 0, "Нет занятых мест"
        elif occupied_count <= 3:
            return random.randint(2, 5), "Скоро освободятся"
        elif occupied_count <= 6:
            return random.randint(5, 10), "В ближайшее время"
        elif occupied_count <= 10:
            return random.randint(8, 15), "Ожидайте"
        else:
            return random.randint(12, 20), "Высокая загрузка"
    
    def predict_hourly_occupancy(self, current_rate):
        forecasts = []
        for i in range(4):
            decay = 0.8 ** (i + 1)
            predicted = max(5, min(95, current_rate * decay + random.uniform(-8, 8)))
            forecasts.append(int(predicted))
        return forecasts
    
    def update_parking_status(self):
        try:
            total = len(self.parking_spots)
            occupied = 0
            disabled_occupied = 0
            disabled_total = 0
            
            for spot in self.parking_spots:
                if spot.spot_type == "disabled":
                    disabled_total += 1
                if random.random() > 0.55:
                    spot.occupied = True
                    occupied += 1
                    if spot.spot_type == "disabled":
                        disabled_occupied += 1
                    spot.free_time = random.randint(5, 45)
                else:
                    spot.occupied = False
                    spot.free_time = None
            
            free = total - occupied
            occupancy_rate = (occupied / total * 100) if total > 0 else 0
            disabled_free = disabled_total - disabled_occupied
            
            # Обновляем статистику
            self.total_label.config(text=str(total))
            self.occupied_label.config(text=str(occupied))
            self.free_label.config(text=str(free))
            self.occupancy_percent.config(text=f"{occupancy_rate:.1f}%")
            self.occupancy_bar['value'] = occupancy_rate
            self.disabled_label.config(text=f"{disabled_free} из {disabled_total} свободно")
            
            # График
            self.occupancy_history.append(occupancy_rate)
            self.update_graph()
            
            # Прогнозы
            full_free = self.predict_full_free_time(occupancy_rate)
            self.full_free_time.config(text=full_free)
            
            next_free_min, next_free_msg = self.predict_next_free(occupied)
            if next_free_min > 0:
                self.next_free_label.config(text=f"{next_free_min} минут")
                self.prediction_message.config(text=f"{next_free_msg}")
            else:
                self.next_free_label.config(text="Все свободны!")
                self.prediction_message.config(text="🎉 Отличное время!")
            
            # Почасовой прогноз
            hourly = self.predict_hourly_occupancy(occupancy_rate)
            for i, label in enumerate(self.hourly_labels):
                if i < len(hourly):
                    label.config(text=f"{hourly[i]}%")
            
            # Обновляем рекомендации
            self.update_tips(occupancy_rate, disabled_free)
            
            # Отрисовка
            self.draw_parking()
            
            now = datetime.now()
            self.status_bar.config(
                text=f"✅ Обновлено: {now.strftime('%H:%M:%S')} | Загрузка: {occupancy_rate:.1f}% | Уведомлений: {self.notification_count}"
            )
            
        except Exception as e:
            self.status_bar.config(text=f"❌ Ошибка: {str(e)}")
    
    def update_tips(self, occupancy_rate, disabled_free):
        """Обновление текста рекомендаций"""
        tips = []
        
        if occupancy_rate < 30:
            tips.append("✅ Много свободных мест")
            tips.append("📍 Рекомендуем парковаться в любом ряду")
        elif occupancy_rate < 70:
            tips.append("🟡 Средняя загрузка")
            tips.append("🎯 Рекомендуем центральный ряд")
            tips.append("🚗 Места у выезда загружены меньше")
        else:
            tips.append("🔴 Высокая загрузка")
            tips.append("⏰ Ожидайте освобождения через 10-20 мин")
            tips.append("♿ Обратите внимание на места для инвалидов")
        
        if disabled_free > 0:
            tips.append(f"♿ Доступно {disabled_free} мест для инвалидов")
        
        if occupancy_rate > 80:
            tips.append("📱 Включите уведомления о свободных местах")
        
        self.tips_label.config(text="\n".join(tips))
            
    def update_graph(self):
        if self.occupancy_history:
            data = list(self.occupancy_history)
            x = range(len(data))
            self.line.set_data(x, data)
            self.ax.set_xlim(0, max(10, len(data)))
            self.ax.set_ylim(0, 100)
            self.ax.fill_between(x, data, alpha=0.3, color=Colors.PRIMARY)
            self.canvas_graph.draw()
        
    def draw_parking(self):
        """Отрисовка парковки"""
        self.canvas.delete("all")
        
        # Фон
        self.canvas.create_rectangle(0, 0, 850, 700, fill='#e8e8e8', outline='')
        
        for spot in self.parking_spots:
            x, y = spot.x, spot.y
            w, h = spot.width, spot.height
            
            if spot.occupied:
                fill_color = Colors.DANGER
                icon = "🚗"
            else:
                fill_color = Colors.SUCCESS
                icon = "🚙"
            
            if spot.spot_type == "disabled":
                border_color = '#2980b9'
                icon = "♿" if not spot.occupied else "🚗♿"
            else:
                border_color = '#27ae60' if not spot.occupied else '#c0392b'
            
            # Тень
            self.canvas.create_rectangle(x+2, y+2, x+w+2, y+h+2, fill='#cccccc', outline='')
            
            # Основной прямоугольник
            self.canvas.create_rectangle(x, y, x+w, y+h, fill=fill_color, outline=border_color, width=2)
            
            # Номер
            self.canvas.create_text(x + w//2, y + 18, text=f"#{spot.id + 1}", 
                                    font=('Segoe UI', 9, 'bold'), fill='white')
            
            # Иконка
            self.canvas.create_text(x + w//2, y + h//2, text=icon, font=('Segoe UI', 20), fill='white')
            
            # Статус
            status = "ЗАНЯТО" if spot.occupied else "СВОБОДНО"
            self.canvas.create_text(x + w//2, y + h - 20, text=status, font=('Segoe UI', 8), fill='white')
            
            # Прогноз
            if spot.occupied and spot.free_time:
                self.canvas.create_text(x + w//2, y + h - 8, text=f"~{spot.free_time}мин", 
                                        font=('Segoe UI', 7), fill='white')
        
        # Зона для инвалидов
        self.canvas.create_text(170, 115, text="♿ МЕСТА ДЛЯ", font=('Segoe UI', 9, 'bold'), fill=Colors.TEXT)
        self.canvas.create_text(170, 130, text="ИНВАЛИДОВ", font=('Segoe UI', 9, 'bold'), fill=Colors.TEXT)
        
        # Легенда
        legend_y = 650
        self.canvas.create_rectangle(100, legend_y, 120, legend_y+20, fill=Colors.SUCCESS, outline='')
        self.canvas.create_text(130, legend_y+10, text="Свободно", anchor='w', font=('Segoe UI', 9))
        
        self.canvas.create_rectangle(240, legend_y, 260, legend_y+20, fill=Colors.DANGER, outline='')
        self.canvas.create_text(270, legend_y+10, text="Занято", anchor='w', font=('Segoe UI', 9))
        
        self.canvas.create_rectangle(380, legend_y, 400, legend_y+20, fill='#3498db', outline='')
        self.canvas.create_text(410, legend_y+10, text="Для инвалидов", anchor='w', font=('Segoe UI', 9))
        
        # Рисуем дорожную разметку
        self.canvas.create_line(250, 70, 250, 630, fill='#95a5a6', width=3, dash=(5, 5))
        self.canvas.create_line(570, 70, 570, 630, fill='#95a5a6', width=3, dash=(5, 5))
        
    def reset_parking(self):
        for spot in self.parking_spots:
            spot.occupied = False
            spot.free_time = None
        self.update_parking_status()
        messagebox.showinfo("Сброс", "Статус парковки сброшен")
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = AdvancedParkingVisualizer()
    app.run()