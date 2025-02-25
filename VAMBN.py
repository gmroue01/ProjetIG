import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

"""
Paramètre	Valeur initiale	Ajustements possibles
Latent dim (latent_dim)	10-15	Augmenter si trop proche, réduire si bruité
Beta (β)	0.1 - 0.2	Diminuer si trop similaire, augmenter si incohérent
Epochs	100	Augmenter si sous-apprentissage, réduire si perte stagne
"""

# Charger les données
print("Chargement des données...")
df = pd.read_csv('Big_Data_Strokdem.csv')

# Supprimer les espaces supplémentaires des noms de colonnes
df.columns = df.columns.str.strip()

# Afficher toutes les colonnes
# print(df.columns.tolist())

print("Prétraitement des données...")
# Sélectionner les colonnes numériques et catégorielles
numeric_features = df.select_dtypes(include=[float, int]).columns.tolist()
categorical_features = df.select_dtypes(include=[object]).columns.tolist()

# Retirer la colonne cible de la liste des features si elle est présente
target = 'MMSE_M6'  # Exemple de colonne cible
if target in numeric_features:
    numeric_features.remove(target)

# Encoder les colonnes catégorielles
df_encoded = pd.get_dummies(df[categorical_features])

# Combiner les colonnes numériques et encodées
data = pd.concat([df[numeric_features], df_encoded], axis=1).values
labels = df[target].values

# Gérer les valeurs manquantes
data = pd.DataFrame(data).fillna(0).values
labels = pd.Series(labels).fillna(0).values

# Normaliser les données
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Diviser les données en ensembles d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Adapter les dimensions d'entrée et de sortie du modèle
input_shape = (x_train.shape[1],)
output_dim = 1  # Exemple pour une sortie scalaire

latent_dim = 25  # Taille de l'espace latent
"""
Si les données générées sont trop similaires → Augmente latent_dim à 20-25.
Si elles sont trop bruitées ou incohérentes → Diminue latent_dim à 5-8.
"""

class VAE(keras.Model):
    """ Modèle VAE """
    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        # Encodeur (4 couches)
        self.encoder = keras.Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(2 * latent_dim),  # [moyenne, log(variance)]
            layers.Dropout(0.3)
        ])

        # Décodeur 4 couches
        self.decoder = keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(input_shape[0], activation="sigmoid")
        ])

    def reparameterize(self, mean, logvar):
        """ Application du trick de reparamétrisation """
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.exp(0.5 * logvar) * eps

    def call(self, x):
        """ Forward pass """
        z_params = self.encoder(x)
        mean, logvar = tf.split(z_params, num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        return self.decoder(z), mean, logvar
    
class BayesianNetwork(PyroModule):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](latent_dim, 32)
        self.fc2 = PyroModule[nn.Linear](32, latent_dim)

    def forward(self, z):
        """ Propagation des variables latentes dans le réseau bayésien """
        z = torch.relu(self.fc1(z))
        return self.fc2(z)

class LSTMDecoder(keras.Model):
    def __init__(self, latent_dim, output_dim):
        super(LSTMDecoder, self).__init__()
        self.lstm = layers.LSTM(64, return_sequences=True, return_state=True)
        self.dense = layers.Dense(output_dim, activation="sigmoid")

    def call(self, x):
        lstm_out, state_h, state_c = self.lstm(x)
        return self.dense(lstm_out)

vae = VAE(latent_dim)
bn = BayesianNetwork(latent_dim)
lstm_decoder = LSTMDecoder(latent_dim, output_dim=output_dim)

# Compilation du modèle
print("Compilation du modèle...")
optimizer = keras.optimizers.Adam(learning_rate=0.001)

import tensorflow as tf

def loss_function(x, recon_x, mean, logvar):
    """ Fonction de perte avec KL divergence """
    beta = 5  # Augmenté pour plus de diversité

    # Utilisation correcte de la perte MSE
    mse_loss = tf.keras.losses.MeanSquaredError()
    recon_loss = tf.reduce_mean(mse_loss(x, recon_x))

    # Calcul de la divergence KL
    kl_loss = -0.5 * tf.reduce_mean(1 + logvar - tf.square(mean) - tf.exp(logvar))

    # Perte totale avec coefficient beta sur KL
    total_loss = recon_loss + beta * kl_loss

    return total_loss


# Entraînement
epochs = 100

#liste pour stocker les pertes
train_losses = []
test_losses = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        recon_x, mean, logvar = vae(x_train)
        loss = loss_function(x_train, recon_x, mean, logvar)
    
    grads = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(grads, vae.trainable_variables))

    # Calculer la perte sur le test
    recon_x_test, mean_test, logvar_test = vae(x_test)
    test_loss = loss_function(x_test, recon_x_test, mean_test, logvar_test)

    # Stocker les valeurs
    train_losses.append(loss.numpy())
    test_losses.append(test_loss.numpy())

    print(f"Epoch {epoch+1}, Train Loss: {loss.numpy():.4f}, Test Loss: {test_loss.numpy():.4f}")


# Tracer la courbe des pertes
plt.figure(figsize=(8,5))
plt.plot(range(1, epochs+1), train_losses, marker='o', linestyle='-', color='b', label='Train Loss')
plt.plot(range(1, epochs+1), test_losses, marker='o', linestyle='-', color='r', label='Test Loss')
plt.title('Évolution de la perte pendant l\'entraînement')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Évaluation sur l'ensemble de test
recon_x, mean, logvar = vae(x_test)
test_loss = loss_function(x_test, recon_x, mean, logvar)
print(f"Test Loss: {test_loss.numpy():.4f}")

# Génération de données synthétiques
print("Génération de données synthétiques...")
def generate_synthetic_data(vae, num_samples):
    z = tf.random.normal(shape=(num_samples, latent_dim))
    synthetic_data = vae.decoder(z)
    return synthetic_data.numpy()

# Générer 1000 échantillons synthétiques
synthetic_data = generate_synthetic_data(vae, 1000)

# Inverser la normalisationx
synthetic_data = scaler.inverse_transform(synthetic_data)

# Convertir les données synthétiques en DataFrame
synthetic_df = pd.DataFrame(synthetic_data, columns=numeric_features + df_encoded.columns.tolist())

# Inverser l'encodage one-hot pour les colonnes catégorielles
for feature in categorical_features:
    # Trouver les colonnes encodées pour cette feature
    encoded_columns = [col for col in synthetic_df.columns if col.startswith(feature + '_')]
    if encoded_columns:
        # Trouver la colonne avec la valeur maximale (one-hot inverse)
        synthetic_df[feature] = synthetic_df[encoded_columns].idxmax(axis=1).str.replace(feature + '_', '')
        # Supprimer les colonnes encodées
        synthetic_df.drop(columns=encoded_columns, inplace=True)

# Sauvegarder les données synthétiques dans un fichier CSV
print("Sauvegarde des données synthétiques...")
synthetic_df.to_csv('synthetic_data.csv', index=False)

print("Données synthétiques générées et sauvegardées dans 'synthetic_data.csv'")