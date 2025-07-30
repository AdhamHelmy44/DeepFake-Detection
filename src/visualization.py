import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd


# (Keep your existing functions here)
# display_final_verdict_banner(...)
# display_frame_verdicts_grid(...)

def plot_training_history(history, method_name, output_dir):
    """Plots and saves the training and validation accuracy and loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Accuracy
    ax1.plot(history['accuracy'], label='Training Accuracy')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'Model Accuracy for {method_name}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Plot Loss
    ax2.plot(history['loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title(f'Model Loss for {method_name}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'history_{method_name}.png'))
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, method_name, output_dir):
    """Plots and saves the confusion matrix."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f'Confusion Matrix for {method_name}')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{method_name}.png'))
    plt.show()


def plot_performance_comparison(df_results):
    """
    Creates a bar chart to compare model performance based on F1-Score.

    Args:
        df_results (pd.DataFrame): DataFrame containing model names and their metrics.
    """
    if df_results is None or df_results.empty:
        print("No results to plot.")
        return

    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(
        data=df_results,
        x='F1-Score',
        y='Model',
        palette='viridis',
        orient='h'
    )

    # Add labels to the bars
    for i, (value, name) in enumerate(zip(df_results['F1-Score'], df_results['Model'])):
        barplot.text(value + 0.01, i, f"{value:.3f}", color='black', ha="left", va='center')

    plt.title('Model Performance Comparison (F1-Score)', fontsize=16, weight='bold')
    plt.xlabel('F1-Score', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xlim(0, 1.1)  # Set x-axis limit from 0 to 1.1 for padding
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()