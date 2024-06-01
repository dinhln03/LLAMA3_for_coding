def neighbord_analysis(x_as, column = 0):
	"""
	Given an array xas this function compute the distance between the elements the mean distance and the variance
	
	Author: Michele Monti
	
	Args:
			x_as: the name of the list or data set that you want:
	
	Kwargs:
		column: is the column of the data set that you need to analyze

	Returns:
		mean_distance: the mean distance between neighbords, 
		std_dev: stdeviation of the distances between neighbords.
		diff_neighbor: the difference between the first-neighbours in a list 
	"""
	
	x_as = np.array(x_as)
	correct_axis = x_as
	if shape(x_as) > 1:
		correct_axis = x_as[:,column]
	
	diff_neighbor = [itm - correct_axis[idx - 1] for idx, itm in enumerate(correct_axis)][1:]
	mean_distance = np.mean(diff_neighbor)
	std_dev = np.std(diff_neighbor)
	
	return(diff_neighbor, mean_distance, std_dev)
