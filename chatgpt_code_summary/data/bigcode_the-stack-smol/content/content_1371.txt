# Configuration file with default options,
# There are four main sections: General, Features, LQP and Learning corresponding to different 
# functionalities. You can disable any of the Features or Learning section (by commenting it out) according to your requirement.

[General] 
# general options
idir=/home/hussain/datasets/LFW/lfwa # images directory path
odir=/scratch/testing/new-experiments/ # path where cropped_images, learned model and computed features will be stored
dataset=LFW # name of dataset to use; it can be either LFW or FERET [currently not supported]
width=80 # width of cropped images
height=150 # height of cropped images
padding=10 # (same as cellsize) use a padding of one cell on each side. This value must be same as the option cell-size has in the features section
xoffset=1 # offsets to be added (from the center position) to the crop window placed over the original aligned images
yoffset=-4
cbdataset=train-val # complete # This option is used only with LQP Features. It is used to choose subset of dataset for codebook learning e.g. in case of LFW it can be either view1 training validation ('train-val') subset or complete view1 set('complete') 
ftype=LQP # Feature types. Choice can be LBP, LTP, LBP+LTP or LQP
usergb=False # if color images, use  color information during feature computations.

[Features]
# options for feature computation
listfile=""  # a list file containing list of cropped images to  compute features
cellsize=10 # cellsize for the histogram grid
tol=5 # [5,7] # tolerance values used for LTP or LQP features (can pass a list, i.e. tol=[5, 7])
  
[LQP] #LQP Options
lqptype=2 # LQP type represent LQP geometric structure. 
#               Choices can be either Disk (2) or Hor+Ver+Diag+ADiag (9) strip.
lqpsize=7 # LQP size represent radius (length of strip) 
#               of LQP disk (HVDA strip) (can pass a list i.e. lqpsize=[5,7])
coding=4 # LQP encoding type can be: Binary (0), Ternary (1) or Split-Ternary (4)
cbsize=150 # Codebook size (number of visual words) used for
#                     LQP computation (can pass a list, i.e. cbsize=[100, 150]
cbfile="" # [Optional] A list file containing list of images for learning the codebook
                   
[Learning] 
# options for model learning
view=complete # view2 # complete # Choice of the dataset, options cans be view1: used for
#                      parameter tuning purposes; view2: used only for model
#                        evaluation; complete: a model parameters will be first
#                      tuned on view1 and results will be reported on view2
ttype=with-pca # Choice of Training with or without PCA (for feature
#                      evaluation) Available options are with-pca or without-
#                        (a pca model is learned and features are compared in the pca space)
#			 or without-pca (features are compared in there original space)
featdir=""  #  Directory path where computed features have been stored,  used if 
#                        learning is being done without feature computation cycle.
dist=cosine #  Distance metric for comparing features. Choices are cosine, chi-square and L2.
# For optimal results use cosine metric for comparing PCA reduced features and 
# chi-squared for comparing non-reduced ones. 
pcadim=[100,  200,  300,  400,  500,  600,  700,  800,  900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000] #  Number of PCA components. You can pass a scalar or list, i.e.
#                        pcadim= 500. In case of a list, all the dimensions will be used
#                        for model learning (on view1) and finally only the best performing one will be 
#                        kept. Note that a single model with max(pcadim) is learned in this case
#                        but evaluation is done using all the dimensions.
# 			 Caution: providing a much higher dimension makes the learning slow and memory
#                        intensive

