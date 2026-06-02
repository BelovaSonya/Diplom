import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import config
import os

class ResultsVisualizer:
    """Класс для визуализации результатов"""
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
        """Построение матрицы ошибок"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Свободно', 'Занято'],
                    yticklabels=['Свободно', 'Занято'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Сохранение
        filename = os.path.join(config.RESULTS_DIR, f"{title.replace(' ', '_')}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"График сохранен: {filename}")
    
    @staticmethod
    def plot_training_history(history, model_name):
        """Визуализация истории обучения"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # График accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title(f'{model_name} - Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # График loss
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title(f'{model_name} - Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        # Сохранение
        filename = os.path.join(config.RESULTS_DIR, f"{model_name}_training.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"График сохранен: {filename}")
    
    @staticmethod
    def plot_feature_importance(model, feature_names, top_n=20):
        """Визуализация важности признаков"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            indices = np.argsort(importance)[::-1][:top_n]
            
            plt.figure(figsize=(10, 6))
            plt.title('Feature Importance (Top {})'.format(top_n))
            plt.bar(range(top_n), importance[indices])
            plt.xticks(range(top_n), [feature_names[i] for i in indices], 
                      rotation=45, ha='right')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.tight_layout()
            
            # Сохранение
            filename = os.path.join(config.RESULTS_DIR, 'feature_importance.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"График сохранен: {filename}")