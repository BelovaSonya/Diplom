# test_load.py
from data_loader import PKLotDataLoader
import config

print("="*60)
print("ТЕСТ ЗАГРУЗКИ ДАННЫХ")
print("="*60)

# Создаем загрузчик
loader = PKLotDataLoader()

# Загружаем данные
X_images, y = loader.load_data_from_structure()

print(f"\nРезультат:")
print(f"Загружено изображений: {len(X_images)}")
print(f"Меток: {len(y)}")

if len(X_images) > 0:
    print(f"Размер изображений: {X_images[0].shape}")
    print(f"Тип данных: {X_images[0].dtype}")
    print(f"Диапазон значений: [{X_images[0].min():.2f}, {X_images[0].max():.2f}]")
    
    # Извлекаем признаки
    features = loader.extract_features_from_images()
    print(f"\nПризнаки извлечены: {features.shape}")