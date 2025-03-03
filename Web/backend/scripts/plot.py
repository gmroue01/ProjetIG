import matplotlib.pyplot as plt
import numpy as np

# Générer des données
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Créer le graphique
plt.figure(figsize=(6,4))
plt.plot(x, y, label="sin(x)", color="teal")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Graphique Matplotlib")
plt.legend()
plt.grid(True)

# Sauvegarder en image
plt.savefig("plot.png", format="png")
plt.close()
