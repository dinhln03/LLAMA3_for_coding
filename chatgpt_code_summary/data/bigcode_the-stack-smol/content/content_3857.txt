#Importing Libraries 
import os
import csv
import sys, getopt

import uuid

import SimpleITK as sitk
import cv2 
import numpy as np
import tensorflow as tf
from flask import Flask, flash, request, redirect, render_template
from flask import jsonify
from flask import send_from_directory
from flask_materialize import Material
from tensorflow.python.keras.backend import set_session
from werkzeug.utils import secure_filename
import shutil
import nibabel as nib
import pandas as pd
import numpy
from sarcopenia_ai.apps.segmentation.segloader import preprocess_test_image





from sarcopenia_ai.apps.server import settings
from sarcopenia_ai.apps.slice_detection.predict import parse_inputs, to256
from sarcopenia_ai.apps.slice_detection.utils import decode_slice_detection_prediction, \
    preprocess_sitk_image_for_slice_detection, adjust_detected_position_spacing, place_line_on_img
from sarcopenia_ai.core.model_wrapper import BaseModelWrapper
from sarcopenia_ai.io import load_image
from sarcopenia_ai.preprocessing.preprocessing import blend2d
from sarcopenia_ai.utils import compute_muscle_area, compute_muscle_attenuation


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
graph = tf.get_default_graph()


import cv2
import numpy as np
def normalise_zero_one(image, eps=1e-8):
    print("Here 1")
    image = image.astype(np.float32)
    ret = (image - np.min(image))
    ret /= (np.max(image) - np.min(image) + eps)
    return ret

def normalise_one_one(image):
    print("Here 2")
    ret = normalise_zero_one(image)
    ret *= 2.
    ret -= 1.
    return ret
    
def preprocess_test_image(image):
    print("Here")
    #image = normalise_one_one(image, -250, 250)
    image = normalise_one_one(image)
    return image
##################    


def find_max(img):
    return np.unravel_index(np.argmax(img, axis=None), img.shape)[0]

#Read arguments
#############################
import argparse 
  
msg = "Adding description"
      
# Initialize parser 
parser = argparse.ArgumentParser(description = msg) 

# Reading the input arguments 
parser.add_argument("-i", "--Input", help = "Input file or folder") 
parser.add_argument('-test_name', type=str, default='Test')

# Read arguments from command line 
args = parser.parse_args() 
  


path = args.Input
test_name = args.test_name

#Creating the result structure variables
main = os.getcwd()
directory = os.path.join(main+'/NII_Data/'+path)


if not os.path.exists(main+'/Results/'+path+"/"):
  os.mkdir(main+'/Results/'+path+'/')

out = os.path.join(main+'/Results/'+path+"/"+test_name+'/')

if os.path.exists(out):
  shutil.rmtree(out)
  os.mkdir(out)

if not os.path.exists(out):
  os.mkdir(out)

out_yes = os.path.join(out+'/Yes')
if not os.path.exists(out_yes):
  os.mkdir(out_yes)

out_no = os.path.join(out+'/No')
if not os.path.exists(out_no):
  os.mkdir(out_no)

out_rev = os.path.join(out+'/Review/')
if not os.path.exists(out_rev):
  os.mkdir(out_rev)

out_csv = os.path.join(out+'/Pred CSVs/')
if not os.path.exists(out_csv):
  os.mkdir(out_csv)


#Load the sarcopenia-ai models 
#set_session(sess)
model_wrapper = BaseModelWrapper(settings.SLICE_DETECTION_MODEL_PATH)
model_wrapper.setup_model()
global slice_detection_model
slice_detection_model= model_wrapper.model
slice_detection_model._make_predict_function()

global segmentation_model
model_wrapper = BaseModelWrapper(settings.SEGMENTATION_MODEL_PATH)
model_wrapper.setup_model()
segmentation_model = model_wrapper.model
segmentation_model._make_predict_function()

####Updated functions to replace older versions listed in the sarcopenia-ai enviroment
#Previous research indicates adjusting the HU range can help bone appear better 
def reduce_hu_intensity_range(img, minv=100, maxv=1500):
    img = np.clip(img, minv, maxv)
    img = 255 * normalise_zero_one(img)

    return img



#Setting up the output file name & Prediction counter 
pred_id = 0
cols = ['Folder_Path','Patient_Folder','Study_Folder','Serie_Folder','L3_detection','L3_position','Total_slices','Confidence','Slice_Thickness', 'Orientation']
lst = []

#Looping through the input folder and analyzing the images 
for folder in os.listdir(directory):
  #Patient Folder
    if(folder=='.DS_Store'):
        continue
  #Study Folder 
    for sub_folder in os.listdir(directory+"/"+folder):
        if(sub_folder=='.DS_Store'):
            continue
  #Series Folder 
        for sub_sub_folder in os.listdir(directory+"/"+folder+"/"+sub_folder):
  #Image Level 
            for file in os.listdir(directory+"/"+folder+"/"+sub_folder+"/"+sub_sub_folder):
                print("IN SUB-SUB-FOLDER: "+sub_sub_folder)
                #print(file)
                if(file.endswith(".nii.gz") or file.endswith(".nii")):
                    print("Processing file: "+file)
                    try:
        
                        if(sub_sub_folder=='.DS_Store'):
                            continue
                        print("IN SUB-SUB-FOLDER: "+sub_sub_folder)
            
                        image_path = directory+"/"+folder+"/"+sub_folder+"/"+sub_sub_folder+"/"+file
                        
                        prob_threshold_U=settings.THRESHOLD_U
                        prob_threshold_L=settings.THRESHOLD_L

                        #Gathering image name 
                        import ntpath
                        head, tail =  ntpath.split(image_path)
                        image_name =  tail or ntpath.basename(head)
            
                        pred_id = pred_id +1
                        print("ID --> "+str(pred_id))
                        results = {"success": False, "prediction": {'id': pred_id}} 
                        sitk_image, _ = load_image(image_path)   
                        print("-----------------------------image path: "+image_path )
                        





                        #The code is not set up to analyze 4 dimensional data.
                        if len(sitk_image.GetSize()) == 4:
                          print("-------- 4D Image: Grabbing only first volume")
                          sitk_image = sitk_image[:, :, :, 0]




                        #Getting image orientation information for output file. 
                        print('-------------- NIB')
                        nib_image = nib.load(image_path)
                        orient_nib=nib.orientations.aff2axcodes(nib_image.affine)

                        print('-------------- Preprocess')
                        #Preprocessing the image 
                        image2d, image2d_preview= preprocess_sitk_image_for_slice_detection(sitk_image)
                        image3d = sitk.GetArrayFromImage(sitk_image)
                        
                        #print(image3d.shape)
                        #print(image2d.shape)
                        #print(image2d_preview.shape)
            
                        spacing = sitk_image.GetSpacing()
                        size = list(sitk_image.GetSize())
                        
                        slice_thickness = spacing[2]
                        


                        #Utilizing the sarcopenia-ai model to predict the L3 vertabrae 
                        with graph.as_default():
                            set_session(sess)
                            preds = slice_detection_model.predict(image2d)

                            
                        print('-------------- Predict')
                        #Processing the model output 
                        pred_z, prob = decode_slice_detection_prediction(preds)
                        slice_z = adjust_detected_position_spacing(pred_z, spacing)
                        print('Prob:   '+ str(prob))
                        print('Slice Z:  ' +str(slice_z) )
                        print('{red_z:   '+str(pred_z))

                        
                        #Normalizing the prediction image to be within %28-%47 percent of the body 
                        new_z_calculate = 0
                        new_pred_z = pred_z
                        new_slice_z = slice_z
                        new_prob = prob



                        print('-------------- Normalize')
                        if(slice_z < .27*size[2] or slice_z > .48*size[2]):
                        
                            print("---------------------debug")
                            print(preds.shape)
                            print(preds.shape[1])
                            new_pred_z = find_max(preds[0, int(.27*preds.shape[1]):int(.48*preds.shape[1])])
                            new_pred_z = new_pred_z + int(.27*preds.shape[1]);
                            new_slice_z = adjust_detected_position_spacing(new_pred_z, spacing)
                            print("old position")
                            print(pred_z)
                            print(slice_z)
                            print("new position")
                            print(new_pred_z)
                            print(new_slice_z)
                            new_z_calculate =1;
                            new_prob = float(preds[0,new_pred_z])
                            
 

 
                        ## Outputting prediction data 
                        print('-------------- Predict CSV')
                        preds_reshaped = preds.reshape(preds.shape[0], -1) 
                        numpy.savetxt(out_csv+"PRED_"+str(pred_id)+".csv", preds_reshaped, delimiter=",")



                        #If the prediction for L3 is above the predifined threshold for acceptance 
                        if (new_prob > prob_threshold_U):
                          print('-------------- Above')
                          image = image3d
                          slice_image = image[new_slice_z,:, :]
                          image2dA = place_line_on_img(image2d[0], pred_z, pred_z, r=1)
                          image2dB = place_line_on_img(image2d[0], -new_pred_z, new_pred_z, r=1)

                          cv2.imwrite(out_yes+"/"+str(pred_id)+'_YES_'+image_name+'_SL.jpg', to256(slice_image))
                          cv2.imwrite(out_yes+"/"+str(pred_id)+'_YES_'+image_name+'_FR.jpg', to256(image2dA))
                          cv2.imwrite(out_yes+"/"+str(pred_id)+'_YES_'+image_name+'_FR2.jpg', to256(image2dB))

                          output = [image_path,folder,sub_folder,sub_sub_folder,'YES',new_slice_z,size[2],new_prob,slice_thickness, orient_nib]
            
                          lst.append(output)
                        
                        #Images where the L3 vertabrae was not identified 
                        elif (new_prob <= prob_threshold_L ):
                          print('-------------- No')
                          image = image3d
                          slice_image = image[new_slice_z,:, :]
                          image2dA = place_line_on_img(image2d[0], -pred_z, -pred_z, r=1)
                          image2dB = place_line_on_img(image2d[0], -new_pred_z, -new_pred_z, r=1)

                          cv2.imwrite(out_no+str(pred_id)+'_NO_'+image_name+'_SL.jpg', to256(slice_image))
                          cv2.imwrite(out_no+str(pred_id)+'_NO_'+image_name+'_FR.jpg', to256(image2dA))
                          cv2.imwrite(out_no+str(pred_id)+'_NO_'+image_name+'_FR2.jpg', to256(image2dB))
            
              

                          output = [image_path,folder,sub_folder,sub_sub_folder,'NO',new_slice_z,size[2],new_prob,slice_thickness, orient_nib]
            
                          lst.append(output)

                        #Images where the L3 vertabrae was identified but confidence requirements were not met. 
                        else:
                          print('-------------- Review')                          
                          image = image3d
                          slice_image = image[new_slice_z,:, :]
                      
                          

                          image2dA = place_line_on_img(image2d[0], pred_z, pred_z, r=1)
                          image2dB = place_line_on_img(image2d[0], new_pred_z, new_pred_z, r=1)

              
                          cv2.imwrite(out_rev+str(pred_id)+'_REVIEW_'+image_name+'_SL_'+str(new_slice_z)+'_PROB_'+str(new_prob)+'.jpg', to256(slice_image))
                          cv2.imwrite(out_rev+str(pred_id)+'_REVIEW_'+image_name+'_FR_'+str(slice_z)+'_PROB_'+str(prob)+'.jpg', to256(image2dA))
                          cv2.imwrite(out_rev+str(pred_id)+'_REVIEW_'+image_name+'_FR2_'+str(new_slice_z)+'_PROB_'+str(new_prob)+'.jpg', to256(image2dB))
                            
                          output = [image_path,folder,sub_folder,sub_sub_folder,'REVIEW',new_slice_z,size[2],new_prob,slice_thickness, orient_nib]
                          lst.append(output)

             
                    #Images that error out (e.g. image orientation is incorrect)
                    except:
                        print('-------------- Wrong')
                        print('-------------- ')
                        print('-------------- ')
                        print("Something went wrong - File: "+image_path)
                        print("Unexpected error"+str(sys.exc_info()[0]))
                        output = [image_path,folder,sub_folder,sub_sub_folder,'Error','','','Something went wrong:'+str(sys.exc_info()[1]),'', orient_nib]
                        lst.append(output)
                        

                    
                      
#Outputting the results dataset 
df = pd.DataFrame(lst, columns=cols)
if not os.path.exists('/content/gdrive/MyDrive/L3-Clean/Results/Summaries/'):
  os.mkdir('/content/gdrive/MyDrive/L3-Clean/Results/Summaries/')


df.to_csv('/content/gdrive/MyDrive/L3-Clean/Results/Summaries/'+path+'_'+test_name+".csv")


print('                                                   ')
print('                                                   ')
print('                                                   ')
print(' --------------    PROCESSING COMPLETE  -------------------                         ')

