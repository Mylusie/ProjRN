### Import

import numpy as np





### Usefull functions

#Plot a 20*20 numpy array
# input : array : array to plot
def plot_array(array):
	return

#Generate a 20*20 array containing a rectangle defined by entry parameters. The 
# rectangle is filled
# input : lenght   : lenght of the rectangle (integer)
#		: height   : height of the rectangle (integer)
#		: top_left : list of 2 elements indicating the position of the top left corner
# 						of the rectangle (tuple of 2 integer, line and column)
# output : rectangle : 20*20 numpy array containing the rectangle
# The function check if the data given make a valide configuration 
def gen_array(lenght, height, top_left):
	rectangle = np.array([])
	return rectangle

# Generate valide data require to build a rectangle in a 20*20 array
# output : rectangle : A list of the rectangle lenght, height and top left corner
#						position ([line, column])
def gen_rectangle():
	rectangle = [0, 0, [0,0]] 
	return rectangle

#Save a list of rectangle array in a file
# input : rectangle_array_list : list of 20*20 rectangle arrays
# 			file 			   : Name of the given file
def save_rectangle_array_list(rectangle_array_list, file):
	return


#Load a list of rectangle numpy arrays
# input  : file : path/to/file
# output : rectangle_array_list : list of numpy arrays rectangles
def load_rectangle_array_list(file):
	return rectangle_array_list

# Main body


if __name__ == "__main__":


	exit(0)
