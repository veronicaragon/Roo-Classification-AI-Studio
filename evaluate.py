import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
import os

class ModelEvaluator:
    def __init__(self, save_path='results'):
        """Initialize evaluator with path to save results."""
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def plot_confusion_matrix(self, y_true, y_pred, labels, title='Confusion Matrix'):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plt.savefig(os.path.join(self.save_path, 'confusion_matrix.png'))
        plt.close()
        
        return cm, cm_percent

    def plot_class_distribution(self, y):
        """Plot and save class distribution."""
        plt.figure(figsize=(10, 6))
        class_counts = pd.Series(y).value_counts()
        
        sns.barplot(x=class_counts.index, y=class_counts.values)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        # Save plot
        plt.savefig(os.path.join(self.save_path, 'class_distribution.png'))
        plt.close()
        
        return class_counts

    def generate_classification_report(self, y_true, y_pred, labels):
        """Generate and save classification report."""
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        
        # Save report
        df_report.to_csv(os.path.join(self.save_path, 'classification_report.csv'))
        
        return df_report

    def plot_precision_recall_curve(self, y_true, y_pred_proba, labels):
        """Plot and save precision-recall curves for each class."""
        plt.figure(figsize=(10, 8))
        
        for i, label in enumerate(labels):
            precision, recall, _ = precision_recall_curve(y_true == i, y_pred_proba[:, i])
            avg_precision = average_precision_score(y_true == i, y_pred_proba[:, i])
            
            plt.plot(recall, precision, 
                    label=f'{label} (AP = {avg_precision:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(self.save_path, 'precision_recall_curves.png'))
        plt.close()

    def analyze_errors(self, X, y_true, y_pred, feature_names=None):
        """Analyze and save error patterns."""
        # Find misclassified examples
        errors = y_true != y_pred
        error_indices = np.where(errors)[0]
        
        error_analysis = pd.DataFrame({
            'True_Label': y_true[error_indices],
            'Predicted_Label': y_pred[error_indices],
            'Features': [X[i] for i in error_indices]
        })
        
        # Save error analysis
        error_analysis.to_csv(os.path.join(self.save_path, 'error_analysis.csv'))
        
        return error_analysis

    def evaluate_model(self, model, X_test, y_test, labels):
        """Run complete evaluation pipeline."""
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = None
        
        # Generate all evaluation metrics and plots
        cm, cm_percent = self.plot_confusion_matrix(y_test, y_pred, labels)
        class_dist = self.plot_class_distribution(y_test)
        classification_report = self.generate_classification_report(y_test, y_pred, labels)
        
        if y_pred_proba is not None:
            self.plot_precision_recall_curve(y_test, y_pred_proba, labels)
        
        error_analysis = self.analyze_errors(X
