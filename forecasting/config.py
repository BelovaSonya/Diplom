import os
import sys
import tempfile
import shutil

# Оригинальный путь с русскими буквами
ORIGINAL_BASE_PATH = r"C:\Users\Администратор\Desktop\Учеба\ДИПЛОМ\pklot_data"

# Создаем временную папку с английским путем
TEMP_BASE = os.path.join(tempfile.gettempdir(), "pklot_data_temp")
IMAGES_PATH = os.path.join(TEMP_BASE, "images")

# Параметры модели
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5  # Для быстрого теста
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Параметры для обучения
USE_GPU = False
AUGMENTATION = True

# Пути для сохранения
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Создание директорий
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Конфигурация загружена")
print(f"Оригинальный путь: {ORIGINAL_BASE_PATH}")
print(f"Временный путь: {TEMP_BASE}")