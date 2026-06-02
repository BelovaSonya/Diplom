"""
Запуск визуализатора парковки
Использует сохраненные модели для прогнозирования
"""

import os
import sys

# Проверяем наличие моделей
models_dir = os.path.join(os.path.dirname(__file__), "models")
required_models = ["Random_Forest.pkl", "Simple_CNN.h5", "scaler.pkl"]

print("="*60)
print("ПРОВЕРКА НАЛИЧИЯ МОДЕЛЕЙ")
print("="*60)

missing_models = []
for model in required_models:
    model_path = os.path.join(models_dir, model)
    if os.path.exists(model_path):
        print(f"✓ {model} - найден")
    else:
        print(f"✗ {model} - НЕ НАЙДЕН")
        missing_models.append(model)

if missing_models:
    print("\n⚠ ВНИМАНИЕ: Некоторые модели не найдены!")
    print("Визуализатор будет работать в демонстрационном режиме.")
    print("\nДля полноценной работы сначала обучите модели:")
    print("python main.py")
    print("\nПродолжить в демо-режиме? (y/n): ", end="")
    
    choice = input().lower()
    if choice != 'y':
        sys.exit(0)
else:
    print("\n✅ Все модели найдены! Визуализатор работает в полном режиме.")

print("\n" + "="*60)
print("ЗАПУСК ВИЗУАЛИЗАТОРА ПАРКОВКИ")
print("="*60)
print("\nИнструкция:")
print("1. Откроется окно с картой парковки")
print("2. Зеленые места - свободны, красные - заняты")
print("3. Обновление происходит автоматически каждые 10 секунд")
print("4. На правой панели отображается прогноз освобождения")
print("\nЗакройте окно для выхода из программы")
print("="*60)

# Запускаем визуализатор
from parking_visualizer import ParkingApp

if __name__ == "__main__":
    app = ParkingApp()
    app.run()