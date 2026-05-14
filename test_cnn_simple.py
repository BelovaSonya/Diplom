# test_cnn_simple.py
print("Тест импорта cnn_models...")

try:
    from cnn_models import ParkingCNNModels
    print("✓ Класс ParkingCNNModels успешно импортирован!")
    
    # Проверяем создание экземпляра
    try:
        model = ParkingCNNModels()
        print("✓ Экземпляр класса создан успешно!")
        print(f"  Размер изображений: {model.img_size}")
    except Exception as e:
        print(f"✗ Ошибка при создании экземпляра: {e}")
        
except ImportError as e:
    print(f"✗ Ошибка импорта: {e}")
    
    # Показываем содержимое файла
    print("\nСодержимое cnn_models.py:")
    print("-" * 50)
    with open('cnn_models.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:30], 1):  # Показываем первые 30 строк
            print(f"{i:3}: {line.rstrip()}")
    print("-" * 50)