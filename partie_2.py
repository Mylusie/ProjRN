### Import

import numpy as np
import matplotlib.pyplot as plt




### Usefull functions

#Plot a 20*20 numpy array
# input : array : array to plot
def plot_array(array):
	#binary_r = 0 en noir et 1 en blanc
	#binary = 0 en blanc et 1 en noir
	plt.imshow(array, cmap='binary', vmin=0, vmax=1)  # Utilisation de cmap='binary' pour afficher en noir et blanc
	plt.show()  # Afficher l'image


#Generate a 20*20 array containing a rectangle defined by entry parameters. The 
# rectangle is filled
# input : lenght   : lenght of the rectangle (integer)
#		: height   : height of the rectangle (integer)
#		: top_left : list of 2 elements indicating the position of the top left corner
# 						of the rectangle (tuple of 2 integer, line and column)
# output : rectangle : 20*20 numpy array containing the rectangle
# The function check if the data given make a valide configuration 
def gen_array( length, height, top_left):
	    # Vérification de la validité des paramètres
    if length <= 0 or height <= 0:
        print("Erreur : La longueur et la hauteur doivent être des entiers positifs.")
        return None
    if top_left[0] < 0 or top_left[1] < 0 or top_left[0] + height > 20 or top_left[1] + length > 20:
        print("Erreur : La position du coin supérieur gauche est invalide pour le rectangle 20x20.")
        return None

   # Création d'un tableau 20x20 rempli de zéros
    rectangle = np.zeros((20, 20))
	
    # Remplissage du rectangle dans le tableau
    for i in range(top_left[0], top_left[0] + height):
        for j in range(top_left[1], top_left[1] + length):
            rectangle[i][j] = 1  # Remplissage avec des 1 pour former le rectangle

    return rectangle


# Generate valide data require to build a rectangle in a 20*20 array
# output : rectangle : A list of the rectangle lenght, height and top left corner
#						position ([line, column])
def gen_rectangle():
    # Génération aléatoire de la longueur, de la hauteur et de la position du coin supérieur gauche
    length = np.random.randint(2, 11)  # Longueur du rectangle entre 1 et 9
    height = np.random.randint(2, 11)  # Hauteur du rectangle entre 1 et 9
    top_left_line = np.random.randint(0, 20 - height)  # Ligne du coin supérieur gauche
    top_left_column = np.random.randint(0, 20 - length)  # Colonne du coin supérieur gauche

    rectangle = [length, height, [top_left_line, top_left_column]]
    return rectangle


#Save a list of rectangle array in a file
# input : rectangle_array_list : list of 20*20 rectangle arrays
# 			file 			   : Name of the given file
def save_rectangle_array_list(rectangle_array_list, file):
	# Convertir la liste de tableaux en un tableau NumPy
    rectangle_array_np = np.array(rectangle_array_list)

    # Sauvegarder le tableau NumPy dans un fichier .npy
    np.save(file, rectangle_array_np)



#Load a list of rectangle numpy arrays
# input  : file : path/to/file
# output : rectangle_array_list : list of numpy arrays rectangles
def load_rectangle_array_list(file):
	# Charger le fichier .npy en tant que tableau NumPy
    rectangle_array_np = np.load(file)

    # Convertir le tableau NumPy en une liste de tableaux
    rectangle_array_list = rectangle_array_np.tolist()
    return rectangle_array_list

# Main body


if __name__ == "__main__":
    listrect = []

    # Générer et enregistrer 50 rectangles aléatoires
    rectangles_to_save = []

    for _ in range(100):
        random_rectangle = gen_rectangle()
        length, height, top_left = random_rectangle
        rectangle_array = gen_array(length, height, top_left)
        rectangles_to_save.append(rectangle_array)

    # Enregistrer les rectangles dans un fichier
    save_rectangle_array_list(rectangles_to_save, "100rectangle.npy")

    # Charger les rectangles à partir du fichier
    loaded_rectangles = load_rectangle_array_list("100rectangle.npy")

    # Afficher chaque rectangle chargé
    for rectangle in enumerate(loaded_rectangles, start=1):
        plot_array(rectangle)

    exit(0)
