import matplotlib.pyplot as plt
import numpy as np

def dataset_histogram(type, score, length, type_mapping, flag, color):
    reversed_mapping = {v: k for k, v in type_mapping.items()}
    type_mapped = {reversed_mapping[k]: v for k, v in type.items()}
    if flag == "train":
        title1 = f"Rozkład klas 'Type' w zbiorze treningowym ({length} samples)"
        title2 = f"Rozkład klas 'Score' w zbiorze treningowym ({length} samples)"
    else:
        title1 = f"Rozkład klas 'Type' w zbiorze walidacyjnym ({length} samples)"
        title2 = f"Rozkład klas 'Score' w zbiorze walidacyjnym ({length} samples)"

    fig, axs = plt.subplots(1, 2, figsize=(14, 10))

    axs[0].bar(type_mapped.keys(), type_mapped.values(), color=color, hatch='.')
    axs[0].set_title(title1)
    axs[0].set_xlabel("Type")
    axs[0].set_ylabel("Proporcje")

    axs[1].bar(score.keys(), score.values, color=color)
    axs[1].set_title(title2)
    axs[1].set_xlabel("Score")
    axs[1].set_ylabel("Proporcje")
    axs[1].set_xticks(np.arange(0, 6, 1))

    plt.tight_layout()
    plt.show()

