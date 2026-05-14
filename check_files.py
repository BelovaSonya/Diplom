# check_files.py
import os
import sys

print("Проверка файлов проекта...")
print("="*50)

files_to_check = [
    'config.py',
    'data_loader.py', 
    'traditional_models.py',
    'cnn_models.py',
    'visualizer.py',
    'predictor.py',
    'main.py'
]

for file in files_to_check:
    if os.path.exists(file):
        print(f"✓ {file} - найден")
        # Проверка класса ParkingCNNModels в cnn_models.py
        if file == 'cnn_models.py':
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'class ParkingCNNModels' in content:
                    print("  ✓ Класс ParkingCNNModels найден")
                else:
                    print("  ✗ Класс ParkingCNNModels НЕ НАЙДЕН!")
    else:
        print(f"✗ {file} - НЕ НАЙДЕН!")

print("="*50)
print("\nПроверка импорта...")

try:
    from cnn_models import ParkingCNNModels
    print("✓ Импорт ParkingCNNModels успешен")
except Exception as e:
    print(f"✗ Ошибка импорта: {e}")