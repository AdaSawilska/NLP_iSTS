import matplotlib.pyplot as plt
import numpy as np

# Dane dla zbioru 1 (7496 próbek)
y_type_1 = {
    6.0: 0.525880,
    0.0: 0.280683,
    4.0: 0.068436,
    2.0: 0.047892,
    3.0: 0.046425,
    5.0: 0.025347,
    1.0: 0.005336
}

y_score_1 = {
    0: 0.525080,
    5: 0.280550,
    4: 0.102855,
    3: 0.056564,
    2: 0.029883,
    1: 0.005069
}

# Dane dla zbioru 2 (3103 próbki)
y_type_2 = {
    6.0: 0.534644,
    0.0: 0.269094,
    4.0: 0.070255,
    3.0: 0.046729,
    2.0: 0.046729,
    5.0: 0.025782,
    1.0: 0.006768
}

y_score_2 = {
    0: 0.533999,
    5: 0.270061,
    4: 0.116984,
    3: 0.045762,
    2: 0.028360,
    1: 0.004834
}

# Mapowanie klas Type na etykiety
type_mapping = {
    0.0: 'EQUI',
    1.0: 'OPPO',
    2.0: 'SPE1',
    3.0: 'SPE2',
    4.0: 'SIMI',
    5.0: 'REL',
    6.0: 'NOALI'
}

# Zamiana kluczy na etykiety tekstowe
y_type_1_mapped = {type_mapping[k]: v for k, v in y_type_1.items()}
y_type_2_mapped = {type_mapping[k]: v for k, v in y_type_2.items()}

# Tworzenie wykresów z etykietami
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Histogram dla y_type zbioru 1 z etykietami
axs[0, 0].bar(y_type_1_mapped.keys(), y_type_1_mapped.values(), color='skyblue', hatch='.')
axs[0, 0].set_title("Rozkład klas 'Type' w zbiorze treningowym (7496 samples)")
axs[0, 0].set_xlabel("Type")
axs[0, 0].set_ylabel("Proporcje")

# Histogram dla y_score zbioru 1
axs[0, 1].bar(y_score_1.keys(), y_score_1.values(), color='skyblue')
axs[0, 1].set_title("Rozkład klas 'Score' w zbiorze treningowym (7496 samples)")
axs[0, 1].set_xlabel("Score")
axs[0, 1].set_ylabel("Proporcje")
axs[0, 1].set_xticks(np.arange(0, 6, 1))

# Histogram dla y_type zbioru 2 z etykietami
axs[1, 0].bar(y_type_2_mapped.keys(), y_type_2_mapped.values(), color='salmon', hatch='.')
axs[1, 0].set_title("Rozkład klas 'Type' w zbiorze walidacyjnym (3103 samples)")
axs[1, 0].set_xlabel("Type")
axs[1, 0].set_ylabel("Proportion")

# Histogram dla y_score zbioru 2
axs[1, 1].bar(y_score_2.keys(), y_score_2.values(), color='salmon')
axs[1, 1].set_title("Rozkład klas 'Score' w zbiorze walidacyjnym (3103 samples)")
axs[1, 1].set_xlabel("Score")
axs[1, 1].set_ylabel("Proportion")
axs[1, 1].set_xticks(np.arange(0, 6, 1))

plt.tight_layout()
plt.show()

