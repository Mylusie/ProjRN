from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
import numpy as np
from matplotlib import pyplot as plt

#1.2 Architecture du réseau de neurones
# Création du modèle CNN pour la classification des pixels
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(window_size, window_size, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sortie binaire pour la classification (0 ou 1)

# Compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#1.3 Choix des données d'apprentissage
# Charger l'image en niveaux de gris
image = cv2.imread('cam.tif', 0)

# Appliquer l'opérateur de Canny pour détecter les contours
edges = cv2.Canny(image, 100, 200)  # Les valeurs 100 et 200 sont les seuils min et max

# Afficher l'image originale et les contours détectés
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Image Originale'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Contours avec l\'opérateur de Canny'), plt.xticks([]), plt.yticks([])

plt.show()

#1.4 Apprentissage
fenetres, labels = generer_fenetres_et_labels(images, window_size)  # Générer les fenêtres et les labels

# Entraînement du modèle
model.fit(fenetres_entrainement, labels_entrainement, epochs=10, batch_size=32, validation_data=(fenetres_validation, labels_validation))

#1.5 Test et Validation
# Préparation des données de test (pseudocode)
fenetres_test, labels_test = generer_fenetres_et_labels(images_test, window_size)

# Évaluation du modèle sur les données de test
evaluation = model.evaluate(fenetres_test, labels_test)

# Affichage des performances du modèle
print("Perte (Loss) :", evaluation[0])
print("Précision (Accuracy) :", evaluation[1])