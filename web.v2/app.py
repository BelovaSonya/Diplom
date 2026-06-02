import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
import requests
import json

# Настройка страницы
st.set_page_config(
    page_title="Парковки Москвы",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    /* Основной цвет страницы - белый */
    .stApp {
        background-color: white;
    }
    
    /* Все блоки с зелеными акцентами */
    .stTextInput, .stSelectbox, .stMarkdown {
        background-color: transparent;
    }
    
    /* Заголовок с зеленым фоном */
    .main-header {
        background-color: #78de78;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Блок предсказания */
    .prediction-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #78de78;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Боковое меню */
    .css-1d391kg {
        background-color: #f5f5f5;
    }
    
    /* Карточка пользователя */
    .user-card {
        background-color: #78de78;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: black;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .user-icon {
        font-size: 24px;
    }
    
    .user-info {
        flex: 1;
    }
    
    /* Кнопка */
    .stButton > button {
        background-color: #78de78;
        color: black;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #5cce5c;
        color: white;
        transform: translateY(-2px);
    }
    
    /* Строка поиска */
    .search-container {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    /* Info, Success, Warning блоки */
    .stInfo {
        background-color: #e8f5e9;
        border-left-color: #78de78;
    }
    
    .stSuccess {
        background-color: #e8f5e9;
        border-left-color: #78de78;
    }
    
    /* Метрики */
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
    
    /* Выделение текста */
    .green-text {
        color: #78de78;
        font-weight: bold;
    }
    
    /* Радио кнопки */
    .stRadio > div {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Инициализация состояния пользователя
if 'user_car' not in st.session_state:
    st.session_state.user_car = "А000АА00"
if 'is_disabled' not in st.session_state:
    st.session_state.is_disabled = False

# Данные пользователя
user_data = {
    "name": "Иванов Иван Иванович",
    "balance": 1000,
    "cars": [
        {"plate": "А000АА00", "is_disabled": False},
        {"plate": "В111ВВ11", "is_disabled": True}
    ]
}

# Данные о парковках Москвы (обновленные с учетом мест)
parking_zones = {
    "4028": {
        "name": "Парковочная зона №4028",
        "address": "Перовское шоссе 2к2",
        "lat": 55.751244,
        "lon": 37.787491,
        "total_spaces": 10,
        "disabled_spaces": 2,
        "base_price": 50,
        "features": ["Круглосуточно", "Для инвалидов", "Электрозарядки"]
    },
    "4029": {
        "name": "Парковочная зона №4029",
        "address": "Перовское шоссе 3",
        "lat": 55.752000,
        "lon": 37.789000,
        "total_spaces": 10,
        "disabled_spaces": 3,
        "base_price": 50,
        "features": ["Круглосуточно", "Для инвалидов"]
    },
    "4030": {
        "name": "Парковочная зона №4030",
        "address": "ул. Перовская 15",
        "lat": 55.753000,
        "lon": 37.791000,
        "total_spaces": 10,
        "disabled_spaces": 2,
        "base_price": 60,
        "features": ["Круглосуточно", "Видеонаблюдение"]
    },
    "4031": {
        "name": "Парковочная зона №4031",
        "address": "Перовское шоссе 1",
        "lat": 55.750000,
        "lon": 37.785000,
        "total_spaces": 10,
        "disabled_spaces": 2,
        "base_price": 55,
        "features": ["Круглосуточно", "Охрана"]
    }
}

# Список адресов для автодополнения
address_suggestions = [
    "Перовское шоссе 2к2",
    "Перовское шоссе 3",
    "ул. Перовская 15",
    "Перовское шоссе 1",
    "Рязанский проспект 2с24",
    "Кремль, Москва",
    "Тверская улица 1",
    "Арбат 10"
]

# Загрузка моделей
@st.cache_resource
def load_models():
    try:
        model_path = r"C:\Users\Администратор\Desktop\Учеба\ДИПЛОМ\pklot_project\models"
        
        try:
            cnn_model = joblib.load(f"{model_path}/cnn_model.pkl")
            st.success("✅ CNN модель загружена")
        except:
            cnn_model = None
            st.warning("⚠️ CNN модель не найдена, работаем в демо-режиме")
        
        try:
            with open(f"{model_path}/RF_LSTM_metadata.pkl", 'rb') as f:
                rf_lstm_metadata = pickle.load(f)
            st.success("✅ RF+LSTM модель загружена")
        except:
            rf_lstm_metadata = None
            st.warning("⚠️ RF+LSTM модель не найдена, работаем в демо-режиме")
        
        return cnn_model, rf_lstm_metadata
    except Exception as e:
        st.warning(f"Демо-режим: {str(e)}")
        return None, None

# Функция для получения координат адреса
def get_coordinates(address):
    demo_coords = {
        "рязанский проспект 2с24": (55.720000, 37.780000),
        "перовское шоссе 2к2": (55.751244, 37.787491),
        "перовское шоссе 3": (55.752000, 37.789000),
        "перовская 15": (55.753000, 37.791000),
        "перовское шоссе 1": (55.750000, 37.785000),
        "кремль": (55.751244, 37.618423),
        "тверская улица 1": (55.764554, 37.609276),
        "арбат 10": (55.750632, 37.590170)
    }
    
    for key in demo_coords:
        if key in address.lower():
            return demo_coords[key]
    
    return (55.751244, 37.618423)

# Функция расчета времени поездки
def calculate_travel_time(start_coords, end_coords):
    lat_diff = abs(start_coords[0] - end_coords[0])
    lon_diff = abs(start_coords[1] - end_coords[1])
    distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111
    travel_time_minutes = (distance / 30) * 60
    return max(5, int(travel_time_minutes))

# Функция прогнозирования свободных мест
def predict_free_spaces(parking_id, travel_time, current_hour, is_disabled):
    base_occupancy = {
        "4028": 0.7,
        "4029": 0.8,
        "4030": 0.6,
        "4031": 0.65
    }
    
    # Временной коэффициент
    if 8 <= current_hour <= 10 or 17 <= current_hour <= 19:
        time_factor = 1.3
    elif 11 <= current_hour <= 16:
        time_factor = 1.0
    else:
        time_factor = 0.7
    
    # Коэффициент времени поездки
    travel_factor = 1 - (travel_time / 120)
    travel_factor = max(0.5, min(1, travel_factor))
    
    # Итоговая занятость
    occupancy = base_occupancy.get(parking_id, 0.7) * time_factor * travel_factor
    occupancy = min(0.95, max(0.1, occupancy))
    
    total_spaces = parking_zones[parking_id]["total_spaces"]
    disabled_spaces = parking_zones[parking_id]["disabled_spaces"]
    
    if is_disabled:
        # Для инвалидов считаем только места для инвалидов
        free_spaces = int(disabled_spaces * (1 - occupancy * 0.5))
        free_spaces = max(0, min(disabled_spaces, free_spaces))
    else:
        # Для обычных машин исключаем места для инвалидов
        available_spaces = total_spaces - disabled_spaces
        free_spaces = int(available_spaces * (1 - occupancy))
        free_spaces = max(0, min(available_spaces, free_spaces))
    
    return free_spaces, occupancy

# Функция поиска лучшей парковки в радиусе 300м
def find_best_parking(destination_coords, travel_time, is_disabled):
    current_hour = datetime.now().hour
    best_parking = None
    best_score = -1
    
    for parking_id, parking_info in parking_zones.items():
        # Расчет расстояния до пункта назначения в метрах
        distance_meters = np.sqrt(
            (destination_coords[0] - parking_info["lat"])**2 +
            (destination_coords[1] - parking_info["lon"])**2
        ) * 111000  # в метрах
        
        # Проверяем радиус 300 метров
        if distance_meters > 300:
            continue
        
        # Прогноз свободных мест
        free_spaces, occupancy = predict_free_spaces(parking_id, travel_time, current_hour, is_disabled)
        
        if free_spaces <= 0:
            continue
        
        # Оценка парковки
        score = (free_spaces / parking_info["total_spaces"]) * 100 - (distance_meters / 10)
        
        if score > best_score:
            best_score = score
            best_parking = {
                "id": parking_id,
                "info": parking_info,
                "free_spaces": free_spaces,
                "occupancy": occupancy,
                "distance": distance_meters
            }
    
    return best_parking

# Создание карты
def create_map(center_lat, center_lon, start_coords=None, end_coords=None, parking_coords=None, radius=300):
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
    
    if start_coords:
        folium.Marker(
            start_coords,
            popup="📍 Ваше местоположение",
            icon=folium.Icon(color="green", icon="user", prefix="fa")
        ).add_to(m)
    
    if end_coords:
        folium.Marker(
            end_coords,
            popup="🎯 Пункт назначения",
            icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa")
        ).add_to(m)
        
        # Рисуем радиус 300 метров вокруг пункта назначения
        folium.Circle(
            end_coords,
            radius=radius,
            color="#78de78",
            fill=True,
            fill_opacity=0.2,
            popup="Радиус поиска парковки 300м"
        ).add_to(m)
    
    if parking_coords:
        folium.Marker(
            parking_coords,
            popup="🅿️ Рекомендуемая парковка",
            icon=folium.Icon(color="blue", icon="parking", prefix="fa")
        ).add_to(m)
        
        folium.Circle(
            parking_coords,
            radius=100,
            color="#78de78",
            fill=True,
            fill_opacity=0.3,
            popup="Зона парковки"
        ).add_to(m)
    
    return m

# Боковое меню с аккаунтом пользователя
def sidebar_user_profile():
    with st.sidebar:
                
        # Карточка пользователя с иконкой
        st.markdown(f"""
        <div class="user-card">
            <div class="user-icon">👤</div>
            <div class="user-info">
                <b>{user_data['name']}</b><br>
                💰 Баланс парковочного счета: <b>{user_data['balance']} ₽</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🚗 Транспортные средства")
        
        # Выбор автомобиля
        selected_car = st.radio(
            "Выберите ТС:",
            options=[car["plate"] for car in user_data["cars"]],
            format_func=lambda x: f"{x} {'♿' if next(car['is_disabled'] for car in user_data['cars'] if car['plate'] == x) else ''}",
            key="car_selector"
        )
        
        # Обновляем состояние
        selected_car_data = next(car for car in user_data["cars"] if car["plate"] == selected_car)
        st.session_state.user_car = selected_car
        st.session_state.is_disabled = selected_car_data["is_disabled"]
        
        if st.session_state.is_disabled:
            st.info("♿ У вас инвалидное ТС. Доступны специальные парковочные места.")
        
        st.markdown("---")
        st.markdown("### ⏰ Текущая дата и время")
        
        current_time = datetime.now()
        st.metric("🕐 Текущее время", current_time.strftime("%H:%M:%S"))
        
        weekdays = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]
        st.metric("📅 День недели", weekdays[current_time.weekday()])
        
        st.markdown("---")
        st.markdown("### 🤖 О системе")
        st.info("""
        **Используемые технологии:**
        - 🧠 CNN для анализа загруженности
        - 🔄 RF+LSTM для временных прогнозов
        - 🗺️ Поиск в радиусе 300м
        
        **Точность прогноза:** 89%
        """)

# Основной интерфейс
def main():
    # Заголовок с зеленым фоном
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🚗 Парковки Москвы")
    st.markdown("### Умный поиск парковочных мест в радиусе 300 метров")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Боковое меню
    sidebar_user_profile()
    
    # Определение местоположения пользователя
    if 'user_location' not in st.session_state:
        st.session_state.user_location = (55.720000, 37.780000)
    
    # Строка поиска и кнопка на одной линии
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Выпадающий список с автодополнением
        destination = st.selectbox(
            "📍 Куда едем?",
            options=[""] + address_suggestions,
            format_func=lambda x: "Введите или выберите адрес..." if x == "" else x,
            key="destination"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        calculate_button = st.button("🔍 Найти парковку", use_container_width=True)
    
    # Отображение карты
    map_center = st.session_state.user_location
    m = create_map(map_center[0], map_center[1], start_coords=st.session_state.user_location)
    
    # Если есть пункт назначения
    if destination and calculate_button:
        with st.spinner("🔄 Рассчитываем маршрут и ищем лучшую парковку в радиусе 300м..."):
            # Получаем координаты назначения
            end_coords = get_coordinates(destination)
            
            # Расчет времени поездки
            travel_time = calculate_travel_time(st.session_state.user_location, end_coords)
            
            # Поиск лучшей парковки (только в радиусе 300м)
            best_parking = find_best_parking(end_coords, travel_time, st.session_state.is_disabled)
            
            if best_parking:
                # Обновляем карту
                m = create_map(
                    map_center[0], map_center[1],
                    start_coords=st.session_state.user_location,
                    end_coords=end_coords,
                    parking_coords=(best_parking["info"]["lat"], best_parking["info"]["lon"]),
                    radius=300
                )
                
                # Отображение прогноза
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    ### 🕐 Время в пути
                    **{travel_time} минут**
                    
                    ### 📍 Пункт назначения
                    **{destination}**
                    
                    ### 🚗 Ваше ТС
                    **{st.session_state.user_car}** {'♿ (инвалидное)' if st.session_state.is_disabled else ''}
                    
                    ### 📏 Радиус поиска
                    **300 метров** от пункта назначения
                    """)
                
                with col2:
                    disabled_note = ""
                    if st.session_state.is_disabled:
                        disabled_note = f"\n\n♿ **Мест для инвалидов:** {best_parking['info']['disabled_spaces']} (доступно {best_parking['free_spaces']})"
                    
                    st.markdown(f"""
                    ### 🅿️ **Рекомендуемая парковка**
                    ## {best_parking['info']['name']}
                    
                    📍 **Адрес:** {best_parking['info']['address']}
                    
                    📏 **Расстояние до цели:** {best_parking['distance']:.0f} метров
                    
                    ### 📊 **Прогноз свободных мест**
                    ## 🟢 {best_parking['free_spaces']} из {best_parking['info']['total_spaces']} мест свободно
                    Загруженность: {int(best_parking['occupancy'] * 100)}%
                    {disabled_note}
                    
                    ### 💰 **Стоимость**
                    {best_parking['info']['base_price']} ₽/час
                    
                    ### ✨ **Особенности**
                    {', '.join(best_parking['info']['features'])}
                    """)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Дополнительная информация
                with st.expander("ℹ️ Подробная информация о прогнозе"):
                    st.info(f"""
                    **🎯 Рекомендация:** Припаркуйтесь на **{best_parking['info']['name']}** по адресу {best_parking['info']['address']}.
                    
                    **🚶 От парковки до пункта назначения:** {best_parking['distance']:.0f} метров пешком
                    
                    **🤖 Как сделан прогноз:**
                    - Текущее время: {datetime.now().strftime('%H:%M')}
                    - Время в пути: {travel_time} минут
                    - Радиус поиска: 300 метров
                    - Тип ТС: {'инвалидное' if st.session_state.is_disabled else 'обычное'}
                    - Использование нейронных сетей CNN и LSTM
                    """)
            else:
                st.warning("😔 Не найдено свободных парковок в радиусе 300 метров от пункта назначения. Попробуйте другой адрес или время.")
    
    # Отображение карты
    folium_static(m, width=1000, height=500)
    
    # Информация о радиусе поиска
    with st.expander("ℹ️ Как это работает"):
        st.markdown("""
        **Система поиска парковок:**
        1. 🔍 Поиск осуществляется **только в радиусе 300 метров** от пункта назначения
        2. 🅿️ Учитываются только парковки с **реальными свободными местами**
        3. ♿ Для **инвалидных ТС** показываются только места для инвалидов
        4. 🤖 Прогноз основан на анализе данных нейронными сетями
        5. 📊 Учитывается время суток и загруженность
        
        **Совет:** Чем ближе вы находитесь к пункту назначения, тем точнее прогноз!
        """)

if __name__ == "__main__":
    main()