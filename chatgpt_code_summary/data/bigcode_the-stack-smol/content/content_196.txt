'''

Autor: Gurkirt Singh
Start data: 2nd May 2016
purpose: of this file is to take all .mp4 videos and convert them to jpg images

'''

import numpy as np
import cv2 as cv2
import math,pickle,shutil,os

baseDir = "/mnt/sun-alpha/actnet/";
vidDir = "/mnt/earth-beta/actnet/videos/";
imgDir = "/mnt/sun-alpha/actnet/rgb-images/";
annotPklFile = "../Evaluation/data/actNet200-V1-3.pkl"
#os.mkdir(imgDir)
annotFile = "../anetv13.json"

def getAnnotations():
    with open(annotFile) as f:
        annoData = json.load(f)
    taxonomy = annoData["taxonomy"]
    version = annoData["version"]
    database = annoData["database"]
    print len(database),version,len(taxonomy)
    
def getNumFrames(filename):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened(): 
        print "could not open :",filename
        return -1
    numf = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    return numf

def getVidedInfo(filename):
    
    try:
        cap = cv2.VideoCapture(filename)
    except cv2.error as e:
        print e
        return 0,0,0,0,-1
    if not cap.isOpened(): 
        print "could not open :",filename
        return 0,0,0,0,-1
    numf = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    return numf,width,height,fps,cap

def getsmallestDimto256(width,height):
    if width>=height:
        newH = 256
        newW = int(math.ceil((float(newH)/height)*width))
    else:
        newW = 256
        newH = int(math.ceil((float(newW)/width)*height))
    return newW,newH
def getframelabels(annotations,numf):
    framelabels = np.ones(numf,dtype='uint16')*200;
    for annot in annotations:
        actionId = annot['class']
        startframe = annot['sf']
        endframe = annot['ef']
        framelabels[startframe:endframe] = int(actionId)-1
    return framelabels

def movefiles(storageDir,framelabels,numfs):
    dst = ''
    for ind in range(numfs):
        label = framelabels[ind]
        src = storageDir+str(ind).zfill(5)+".jpg"
        dst = storageDir+str(ind).zfill(5)+'-ActId'+str(label).zfill(3)+'.jpg'
        shutil.move(src,dst)
    print dst ,' MOVED '
def convertVideosL():
    print "this is convertVideos function with labels"
    ecount = 0
    with open(annotPklFile,'rb') as f:
         actNetDB = pickle.load(f)
    actionIDs = actNetDB['actionIDs']; taxonomy=actNetDB['taxonomy']; database = actNetDB['database'];
    for videoId in reversed(database.keys()):
        ecount+=1
        if ecount>0:
            videoInfo = database[videoId]
            storageDir = imgDir+'v_'+videoId+"/"
            print videoInfo['subset'] 
            if not videoInfo['isnull'] and not videoInfo['subset'] == 'testing':
                videoname = vidDir+'v_'+videoId+'.mp4'
                if not os.path.isfile(videoname):
                    videoname = vidDir+'v_'+videoId+'.mkv' 
                print storageDir,' ecount ',ecount,videoInfo['subset']
                numfs = videoInfo['numf']
                annotations = videoInfo['annotations']
                framelabels = getframelabels(annotations,numfs)
                imgname = storageDir+str(numfs-1).zfill(5)+".jpg"
                if os.path.isfile(imgname):
                    movefiles(storageDir,framelabels,numfs)
                else:
                    dst = storageDir+str(numfs-1).zfill(5)+'-ActId'+str(framelabels[-1]).zfill(3)+'.jpg'
                    if not os.path.isfile(dst):
                        numf,width,height,fps,cap = getVidedInfo(videoname)
                        if not cap == -1 and numf == numfs:
                            newW=256;newH=256;
                            framecount = 0;
                            if cap.isOpened():
                                if not os.path.isdir(storageDir):
                                    os.mkdir(storageDir)
                                
                                for ind in xrange(numf):
                                    label = framelabels[ind]
                                    dst = storageDir+str(ind).zfill(5)+'-ActId'+str(label).zfill(3)+'.jpg'
                                    retval,image = cap.read()
                                    if not image is None:
                                        resizedImage = cv2.resize(image,(newW,newH))
                                        cv2.imwrite(dst,resizedImage)
                                    else:
                                        cv2.imwrite(dst,resizedImage)
                                        print ' . ',
                                print dst , 'is created'
                        else:
                            with open('vids/'+videoId+'.txt','wb') as f:
                                f.write('error')
                    else:
                        print dst , 'is already there'
            
                   

def convertTestVideos():
    print "this is convertVideos function with labels"
    ecount = 0
    with open(annotPklFile,'rb') as f:
         actNetDB = pickle.load(f)
    actionIDs = actNetDB['actionIDs']; taxonomy=actNetDB['taxonomy']; database = actNetDB['database'];
    for videoId in database.keys():
        ecount+=1
        if ecount>0:
            videoInfo = database[videoId]
            storageDir = imgDir+'v_'+videoId+"/"
            print videoInfo['subset'] 
            if not videoInfo['isnull'] and videoInfo['subset'] == 'testing':
                videoname = vidDir+'v_'+videoId+'.mp4'
                if not os.path.isfile(videoname):
                    videoname = vidDir+'v_'+videoId+'.mkv' 
                print storageDir,' ecount ',ecount,videoInfo['subset']
                numfs = videoInfo['numf']
                
                # annotations = videoInfo['annotations']
                framelabels = np.ones(numfs,dtype='uint16')*200;
                imgname = storageDir+str(numfs-1).zfill(5)+".jpg"
                if os.path.isfile(imgname):
                    movefiles(storageDir,framelabels,numfs)
                else:
                    dst = storageDir+str(numfs-1).zfill(5)+'-ActId'+str(framelabels[-1]).zfill(3)+'.jpg'
                    if not os.path.isfile(dst):
                        numf,width,height,fps,cap = getVidedInfo(videoname)
                        if not cap == -1 and numf == numfs:
                            newW=256;newH=256;
                            framecount = 0;
                            if cap.isOpened():
                                if not os.path.isdir(storageDir):
                                    os.mkdir(storageDir)
                                
                                for ind in xrange(numf):
                                    label = framelabels[ind]
                                    dst = storageDir+str(ind).zfill(5)+'-ActId'+str(label).zfill(3)+'.jpg'
                                    retval,image = cap.read()
                                    if not image is None:
                                        resizedImage = cv2.resize(image,(newW,newH))
                                        cv2.imwrite(dst,resizedImage)
                                    else:
                                        cv2.imwrite(dst,resizedImage)
                                        print ' . ',
                                print dst , 'is created'
                        else:
                            with open('vids/'+videoId+'.txt','wb') as f:
                                f.write('error')
                    else:
                        print dst , 'is already there'
           
def convertVideos():
    print "this is convertVideos function"
##    vidDir = vidDirtemp
    vidlist = os.listdir(vidDir)
    vidlist = [vid for vid in vidlist if vid.startswith("v_")]
    print "Number of sucessfully donwloaded ",len(vidlist)
    vcount =0
    for videname in reversed(vidlist):
        vcount+=1
        if vcount>0:
            src = vidDir+videname
            numf,width,height,fps,cap = getVidedInfo(src)
            if not cap == -1:
                newW=256;newH=256;
                print videname, width,height,' and newer are ',newW,newH, ' fps ',fps,' numf ', numf, ' vcount  ',vcount
                framecount = 0;
                storageDir = imgDir+videname.split('.')[0]+"/"
                imgname = storageDir+str(numf-1).zfill(5)+".jpg"
                if not os.path.isfile(imgname):
                    if cap.isOpened():
                        if not os.path.isdir(storageDir):
                            os.mkdir(storageDir)
                        for f in xrange(numf):
                            retval,image = cap.read()
                            if not image is None:
                                # print np.shape(retval),np.shape(image), type(image),f
                                resizedImage = cv2.resize(image,(newW,newH))
                                imgname = storageDir+str(framecount).zfill(5)+".jpg"
                                cv2.imwrite(imgname,resizedImage)
                            else:
                                imgname = storageDir+str(framecount).zfill(5)+".jpg"
                                cv2.imwrite(imgname,resizedImage)
                                print 'we have missing frame ',framecount
                            framecount+=1
                        print imgname
            else:
                with open('vids/'+videname.split('.')[0]+'.txt','wb') as f:
                    f.write('error')
            
def getframelabels4both(annotations,numf,subset):
    framelabels = np.ones(numf,dtype='uint16')*200;
    if subset == 'testing':
        return framelabels
    for annot in annotations:
        actionId = annot['class']
        startframe = annot['sf']
        endframe = annot['ef']
        framelabels[startframe:endframe] = int(actionId)-1
    return framelabels

                 
def genVideoImageLists():
    subset = 'testing'
    print "this is genVideoImageLists function"
    ecount = 0; vcount = 0;
    listname = '{}lists/{}.list'.format(baseDir,subset)
    fid = open(listname,'wb')
    with open(annotPklFile,'rb') as f:
         actNetDB = pickle.load(f)
    actionIDs = actNetDB['actionIDs']; taxonomy=actNetDB['taxonomy']; database = actNetDB['database'];
    
    for videoId in database.keys():
        ecount+=1
        if ecount>0:
            videoInfo = database[videoId]
            if not videoInfo['isnull'] and videoInfo['subset'] == subset:
                vcount+=1
                storageDir = imgDir+'v_'+videoId+"/"
                videlistName = '{}lists/{}/v_{}.list'.format(baseDir,subset,videoId)
                fid.write(videlistName+'\n');
                vfid = open(videlistName,'wb');
                print storageDir,' ecount ',ecount,videoInfo['subset']
                numfs = videoInfo['numf']
                annotations = videoInfo['annotations']
                framelabels = getframelabels4both(annotations,numfs,subset)
                dst = storageDir+str(numfs-1).zfill(5)+'-ActId'+str(framelabels[-1]).zfill(3)+'.jpg'
                if os.path.isfile(dst):
                    for ind in xrange(numfs):
                        label = framelabels[ind]
                        dst = storageDir+str(ind).zfill(5)+'-ActId'+str(label).zfill(3)+'.jpg'
                        vfid.write(dst+'\n')
                else:
                    RuntimeError('check if file exists '+dst)
               
def checkConverted():
    print "this is checkConverted videos function"    
    vidlist = os.listdir(vidDir)
    vidlist = [vid for vid in vidlist if vid.endswith(".mp4")]
    print "Number of sucessfully donwloaded ",len(vidlist)
    vcount =0
    for videname in vidlist[15000:]:
        src = vidDir+videname
        numF = getNumFrames(src)
        if numF>0:
            imgname = imgDir+videname.split('.')[0]+"/"+str(numF-1).zfill(5)+".jpg"
            print 'last frame is ',imgname,' vocunt ',vcount
            vcount+=1
            dst = vidDirtemp+videname
            if not os.path.isfile(imgname):
                shutil.move(src,dst)
                print " moved this one to ", dst

if __name__=="__main__":
    # checkConverted()
    # convertVideosL()
    # convertTestVideos()
    genVideoImageLists()
