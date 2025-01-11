import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def prepare_dataframe(df):
    """Prepares dataframe for calculations."""
    type_mapping = {0: 'EQUI', 1: 'OPPO', 2: 'SPE1', 3: 'SPE2',
                    4: 'SIMI', 5: 'REL', 6: 'NOALI'}

    df['predicted_type'] = df['predicted_type'].map(type_mapping)
    df['predicted_score'] = df['predicted_score'].apply(lambda x: 'NIL' if x == 0 else x).astype(str)
    df['y_score'] = df['y_score'].astype(str)
    df['both_correct'] = (df['y_type'] == df['predicted_type']) & (df['y_score'] == df['predicted_score'])

    return df

def calculate_metrics(data):
    """Calculates metrics of accuracy, precision, recall, f1 score for predictions."""
    real_scores = data['y_score']
    predicted_scores = data['predicted_score']

    cm = confusion_matrix(real_scores, predicted_scores, labels=data['y_score'].unique())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data['y_score'].unique())
    fig, ax = plt.subplots(figsize=(9, 8))
    disp.plot(cmap='RdPu', ax=ax)
    plt.show()

    accuracy_scr = accuracy_score(real_scores, predicted_scores)
    precision_scr = precision_score(real_scores, predicted_scores, average='weighted', zero_division=0)
    recall_scr = recall_score(real_scores, predicted_scores, average='weighted', zero_division=0)
    f1_scr = f1_score(real_scores, predicted_scores, average='weighted', zero_division=0)

    real_types = data['y_type']
    predicted_types = data['predicted_type']

    cm = confusion_matrix(real_types, predicted_types, labels=data['y_type'].unique())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data['y_type'].unique())
    fig, ax = plt.subplots(figsize=(9, 8))
    disp.plot(cmap='RdPu', ax=ax)
    ax.set_xticklabels(disp.display_labels, rotation=90)
    plt.show()

    accuracy_type = accuracy_score(real_types, predicted_types)
    precision_type = precision_score(real_types, predicted_types, average='weighted', zero_division=0)
    recall_type = recall_score(real_types, predicted_types, average='weighted', zero_division=0)
    f1_type = f1_score(real_types, predicted_types, average='weighted', zero_division=0)

    overall_accuracy = df['both_correct'].mean()

    results = {
        'Score Metrics': {
            'Accuracy': accuracy_scr,
            'Precision': precision_scr,
            'Recall': recall_scr,
            'F1 Score': f1_scr
        },
        'Type Metrics': {
            'Accuracy': accuracy_type,
            'Precision': precision_type,
            'Recall': recall_type,
            'F1 Score': f1_type
        }
    }

    print("Miary dla wyników (Score):")
    print(results['Score Metrics'])

    print("\nMiary dla typów (Type):")
    print(results['Type Metrics'])

    print(f"Overall Accuracy (Type + Score): {overall_accuracy}")

if __name__ == "__main__":
    csv_path = "results/predictions_test_images.csv"
    df = pd.read_csv(csv_path, sep=',')
    df_prepared = prepare_dataframe(df)
    calculate_metrics(df_prepared)

