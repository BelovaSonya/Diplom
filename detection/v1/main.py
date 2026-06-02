import os
import sys
import numpy as np
import config
from data_loader import PKLotDataLoader
from traditional_models import ParkingTraditionalModels
from cnn_models import ParkingCNNModels
from visualizer import ResultsVisualizer
from predictor import ParkingPredictor

def main():
    """Главная функция для запуска всего пайплайна"""
    
    print("\n" + "="*70)
    print("ПРОГНОЗИРОВАНИЕ ЗАНЯТОСТИ ПАРКОВОЧНЫХ МЕСТ")
    print("="*70)
    
    # 1. Загрузка данных
    print("\nШАГ 1: Загрузка данных")
    data_loader = PKLotDataLoader()
    X_images, y = data_loader.load_data_from_structure()
    
    if len(X_images) == 0:
        print("Ошибка: данные не загружены. Проверьте путь к изображениям.")
        return
    
    # Извлечение признаков
    X_features = data_loader.extract_features_from_images()
    
    # 2. Обучение традиционных моделей
    print("\nШАГ 2: Обучение традиционных моделей ML")
    traditional = ParkingTraditionalModels()
    X_train_scaled, X_test_scaled, y_train, y_test = traditional.train_models(X_features, y)
    
    # Получение предсказаний для визуализации
    y_pred_rf = traditional.models['Random Forest'].predict(X_test_scaled)
    
    # 3. Обучение CNN моделей
    print("\nШАГ 3: Обучение CNN моделей")
    cnn = ParkingCNNModels()
    X_test_cnn, y_test_cnn, cnn_results = cnn.train_models(X_images, y)
    
    # Получение предсказаний для визуализации
    y_pred_cnn_prob = cnn.models['Simple CNN'].predict(X_test_cnn)
    y_pred_cnn = (y_pred_cnn_prob > 0.5).astype(int)
    
    # 4. Визуализация результатов
    print("\nШАГ 4: Визуализация результатов")
    visualizer = ResultsVisualizer()
    
    # Матрица ошибок для Random Forest
    visualizer.plot_confusion_matrix(y_test, y_pred_rf, 
                                     "Random Forest - Confusion Matrix")
    
    # Матрица ошибок для CNN
    visualizer.plot_confusion_matrix(y_test_cnn, y_pred_cnn,
                                     "CNN - Confusion Matrix")
    
    # Графики обучения
    for name, history in cnn.history.items():
        visualizer.plot_training_history(history, name)
    
    # Важность признаков
    feature_names = [f'feature_{i}' for i in range(X_features.shape[1])]
    visualizer.plot_feature_importance(traditional.models['Random Forest'], 
                                       feature_names, top_n=20)
    
    # 5. Сохранение моделей
    print("\nШАГ 5: Сохранение моделей")
    traditional.save_models()
    cnn.save_models()
    
    # 6. Итоговый отчет
    print("\n" + "="*70)
    print("ИТОГОВЫЙ ОТЧЕТ")
    print("="*70)
    
    print("\nСтатистика данных:")
    print(f"  Всего изображений: {len(X_images)}")
    print(f"  Занятых: {sum(y)} ({sum(y)/len(X_images)*100:.1f}%)")
    print(f"  Свободных: {len(y)-sum(y)} ({(len(y)-sum(y))/len(X_images)*100:.1f}%)")
    
    print("\nПроизводительность моделей:")
    rf_accuracy = traditional.models['Random Forest'].score(X_test_scaled, y_test)
    print(f"  Random Forest Accuracy: {rf_accuracy:.4f}")
    
    if 'Simple CNN' in cnn_results:
        print(f"  Simple CNN Accuracy: {cnn_results['Simple CNN']['Accuracy']:.4f}")
    
    best_accuracy = max(rf_accuracy, cnn_results.get('Simple CNN', {}).get('Accuracy', 0))
    best_model = "Random Forest" if rf_accuracy > cnn_results.get('Simple CNN', {}).get('Accuracy', 0) else "Simple CNN"
    
    print(f"\nЛучшая модель: {best_model} с точностью {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    print("\nМодели сохранены в папке:", config.MODELS_DIR)
    print("Графики сохранены в папке:", config.RESULTS_DIR)
    
    print("\n" + "="*70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print("="*70)

def test_prediction():
    """Тестирование на одном изображении"""
    print("\n" + "="*70)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("="*70)
    
    # Загрузка моделей
    traditional_path = os.path.join(config.MODELS_DIR, "Random_Forest.pkl")
    cnn_path = os.path.join(config.MODELS_DIR, "Simple_CNN.h5")
    
    if os.path.exists(traditional_path) and os.path.exists(cnn_path):
        predictor = ParkingPredictor(traditional_path, cnn_path)
        
        # Тестовое изображение
        sunny_path = os.path.join(config.IMAGES_PATH, "sunny", "occupied")
        if os.path.exists(sunny_path):
            images = [f for f in os.listdir(sunny_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                test_image = os.path.join(sunny_path, images[0])
                print(f"\nТестирование на изображении: {test_image}")
                result = predictor.predict(test_image, method='ensemble')
                
                if result:
                    print("\nРезультаты прогнозирования:")
                    for model_name, pred in result.items():
                        print(f"  {model_name}: {pred['label']} (вероятность: {pred['probability']:.3f})")
            else:
                print("Нет изображений в папке sunny/occupied")
        else:
            print(f"Папка не найдена: {sunny_path}")
    else:
        print("Модели не найдены. Сначала запустите обучение (main())")

if __name__ == "__main__":
    # Выбор режима работы
    print("Выберите режим работы:")
    print("1 - Обучение моделей")
    print("2 - Тестирование на одном изображении")
    print("3 - Обучение + Тестирование")
    
    choice = input("Ваш выбор (1/2/3): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        test_prediction()
    elif choice == "3":
        main()
        test_prediction()
    else:
        print("Неверный выбор. Запускается обучение по умолчанию.")
        main()