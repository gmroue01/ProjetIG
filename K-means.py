import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 📌 Charger les données CSV (sans en-tête)
df = pd.read_csv("data.csv", header=None)

# 📌 Centralisation et normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df) #J'exclue le dernier patient avec [:-1]

# 📌 Appliquer K-Means (choisir k=3, mais ajustable)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# 📌 Réduction de dimension avec PCA pour la visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 📌 Création d'un DataFrame pour visualisation
df_pca = pd.DataFrame(X_pca, columns=["PCA1", "PCA2"])
df_pca["Cluster"] = clusters  # Ajouter les clusters

# 📌 Sauvegarde du fichier formaté
df_pca.to_csv("clusters_output.csv", index=False, header=False)

# 📌 Vérification du DataFrame avant affichage
print(df_pca)  # Debug

# 📌 Affichage des clusters en 2D
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="Cluster", palette="viridis", alpha=0.7)
plt.title("Clustering - Projection PCA")
plt.show()

# 📌 Patient test
patient_test = np.array([[0.0,85.0,22.0,48.0,0.0,57.0,48.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,80.0,185.0,23.4,1.0,45.0,91.0,13.0,41.0,81.0,15.0,27.0,27.0,25.0,26.0,24.0,26.0,24.0,25.0,3.0,1.0,1.0,1.0,1.0
]])
patient_test_scaled = scaler.transform(patient_test)
cluster_patient_test = kmeans.predict(patient_test_scaled)

print(f"Cluster du patient test: {cluster_patient_test[0]} PCA: {pca.transform(patient_test_scaled)}")