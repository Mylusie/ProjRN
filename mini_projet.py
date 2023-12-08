###  Import
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import SGD

import numpy as np
import matplotlib.pyplot as plt

import scipy.io as sio

from PIL import Image
from math import floor
import cv2

### Usefull global variables

# List of images in DetectionDeFormes folder Excluded : "blood1"
IMAGES = ["albatross", "cam", "frog", "ImageEntree", "mandrillE", "tools1",
			"bacsE", "campanile", "coinsE", "gecko", "leopard", "roo" ]
# Size of the kernel to generate the thumbnails, must be odd and >=3
KERNEL_SIZE = 3

#NN optimiser
OPTIMIZER = "adam"#SGD(learning_rate=0.01)
 ### Fonctions

# Allow to generate the images database and it's edged with Canny version
# input  : image_list : A list of image names located in DetectionDeFormes folder
# output : a tuble of (0) a list of images and (1) a list of edged images with
#			 Canny's filter
def gen_database(image_list):

	list_images_edge = []
	list_images 	 = []

	for i in IMAGES:
		cur_image = np.array(Image.open("DetectionDeFormes/" + i + ".tif"))
		list_images.append(cur_image)

		cur_image_edge = cv2.Canny(cur_image, 85, 255)

		show_im(cur_image_edge)

		#Normalisation of edged image
		for line in range(len(cur_image_edge)):
			for column in range(len(cur_image_edge[line])):
				if (cur_image_edge[line][column] == 255):
					cur_image_edge[line][column] =  1

		list_images_edge.append(cur_image_edge)

	return (list_images, list_images_edge) 

# Generate the NN dataset
# input  : database : database where (0) is a vector of images and (1) is a vector of
#						those images edges
# output : entry 	: couple were (0) are thumbnails and (1) bits indicating
#						whether the corresponding center should be considered an
#						edge or not
def gen_entry(database):
	entry_th  = [] # What we put in the NN
	entry_res = [] # What is suppose to come out the NN

	### Part to modifie in order to modifie the NN entry
	image_th = gen_thumbnails(database[0][4], KERNEL_SIZE)
	size = len(database[1][4])
	image_edges = np.reshape(database[1][4],size*size)

	for i in range(len(image_th)):
		should_add = filter_th(entry_th, image_th[i])
		if(should_add):
			entry_th.append(image_th[i])
			entry_res.append(image_edges[i])
	### End of the part

	entry = (entry_th, entry_res)
	return entry

#Generate a list of thumbnails of size n*n. Border are 0-padded
# input : image : numpy array of the image to cut 
# 			n 	: The size of a thumbnail, must be odd, >=3
#					and smaller in both dimensions than the image itself (default == 3)
# output : image_th : list of the thumbnails made, which are numpy arrays
def gen_thumbnails(image, n=3):
	image_th = []

	# First we do the zero padding
	n_half = n//2
	for i in range(n_half):
		image = np.insert(image, (0, len(image)), 0, axis=0) # Add a line of 0 at the
											 	#  beginning and at the end
		image = np.insert(image, (0, len(image[0])), 0, axis=1) # Same for columns

	# Generate thumbnails
		# We go through the image skipping the previously padded part
	for line in range(n_half, len(image) - n_half): 
		for column in range(n_half, len(image[0]) - n_half):
			
			# Imagining the thumbnail is on top of the image and we fill it
			#  from top-left corner to bottom-right one, line by line
			#  line_th and column_th then give us the current offset from the center
			#  of the thumbnail point of view
			thumbnail = []
			line_th   = - n_half
			column_th = - n_half
			for k in range(n*n):
				if(column_th > n_half):
					column_th = - n_half
					line_th +=1

				element = image[line+line_th][column+column_th]

				thumbnail.append(element)

				column_th +=1

			image_th.append(np.array(thumbnail))

	return image_th

# Return a boolean indicating if the current thumbnail is worth enought
#  to get into the thumbnail list
# input  : entry_th   : Current list of thumbnail to use, may be empty
#			thumbnail : Thumbnail being tested
# output : A boolean indicating whether or not to add the new one
def filter_th(entry_th, thumbnail):
	return True

 # Try NN over a list of images
 # input : model 	  : The NN model
 #		 : image_list : The list of images to try
 #		 : limit 	  : Value to differentiate a 0 from a 1, must be in [0,1]
 #output : post_NN_list : The list of images through the NN
def try_db(model, image_list, limit):
	post_NN_list = []

	for i in image_list:
		nb_line = len(i)
		nb_column = len(i[0])
		#show_im(i)
		image_th = gen_thumbnails(i, KERNEL_SIZE)

		post_NN = model.predict(np.asarray(image_th))

		for i in range(len(post_NN)):
			if post_NN[i] < limit:
				post_NN[i] = 0
			else:
				post_NN[i] = 1

		post_NN = denormalise(post_NN)
		post_NN = np.asarray(post_NN)
		post_NN_list.append(post_NN.reshape(nb_line,nb_column))

	return post_NN_list

# Take an array image normalised between 0 and 1 and return 
#	it denormalised between 0 and 255
def denormalise(image):

	for line in range(len(image)):
		for column in range(len(image[line])):
			if image[line][column] == 1:
				image[line][column] = 255

	return image

# Plot an image, press 0 to close
# input : image : A numpy array corresponding to the image to show
def show_im(image):

	cv2.imshow("Image", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	
	return

### Exécution
if __name__ == "__main__":

	print("Hello world !")

	datab = gen_database(IMAGES) # database, (0) is a vector of images and
								#   (1) is a vector of those images edged with Canny


	entry = gen_entry(datab) # (0) is a list of entry for the NN and (1) is the 
									# list of what's suppose to get out from the NN


	training_limit = floor(len(entry[0])*0.7) 

	X_train = np.asarray(entry[0][:training_limit])
	Y_train = np.asarray(entry[1][:training_limit])
	X_test  = np.asarray(entry[0][training_limit:])  
	Y_test  = np.asarray(entry[1][training_limit:])

	epochs = 100
	batch_size = len(X_train)

	model = keras.models.Sequential(name="edge_NN")
	model.add(Input(shape=(KERNEL_SIZE*KERNEL_SIZE,)))
	model.add(Dense(5,activation="tanh"))
	model.add(Dense(1, activation="sigmoid"))
	model.compile(optimizer=OPTIMIZER,
              loss     ="binary_crossentropy",
              metrics  =["accuracy"])
	
	history=model.fit(X_train,Y_train, epochs=epochs,
                  batch_size=batch_size, verbose=1)

	taux = model.evaluate(X_test, Y_test)



	images_NN = try_db(model, datab[0], 0.5)

	for i in images_NN:
		show_im(i)


	exit(0)

