import numpy as np
import matplotlib.pyplot as plt
from math import pi

"Tworzony jest wykres radarowy na podstawie wczytywanych danych"

#Wczytywanie danych
scores = np.load("Results/metric_results.npy")
scores = np.mean(scores, axis=2).T
scores = np.mean(scores, axis=3).T
scores = np.mean(scores, axis=2).T



methods = ['none','RUS' ,'CC', 'NM', 'MCC', 'MCC-2', 'MCC-3', 'MCC-4']
metrics = ['specificity','g-mean','bac', 'f1_score']
N = scores.shape[0]

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

ax = plt.subplot(111,polar=True)

# pierwsza os na gorze
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# po jednej osi na metryke
plt.xticks(angles[:-1], metrics)

# os y
ax.set_rlabel_position(0)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
color="grey", size=7)
plt.ylim(0,1)

# Dodajemy wlasciwe ploty dla kazdej z metod
for method_id, method in enumerate(methods):
    values=scores[:, method_id].tolist()
    values += values[:1]
    print(values)
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=method)

# Dodajemy legende
plt.legend(bbox_to_anchor=(0, 0), ncol=8)
# Zapisujemy wykres
plt.savefig("Results/radar", dpi=200)