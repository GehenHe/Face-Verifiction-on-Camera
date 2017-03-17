#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:44:49 2017

@author: gehen
"""

import numpy as np
import cv2
from preprocess import NaiveDlib,Wrapper
from PIL import Image

aligne_model = './model/dlib/shape_predictor_68_face_landmarks.dat'
caffe_model = './model/caffe/FACE.caffemodel'
caffe_proto = './model/caffe/val.prototxt'
mean_file = './model/caffe/mean.npy'
im_dir = '/home/gehen/facesearcher/facesearch1/data/face.jpg'
aligner = NaiveDlib(aligne_model)
extractor = Wrapper(caffe_proto, caffe_model, mean_file)

cap = cv2.VideoCapture(-1)
id_img = cv2.imread('/home/gehen/PycharmProjects/Face_Verification/data/test2/320322198709010001.jpg')
bb = aligner.getAllFaceBoundingBoxes(id_img)
alignedFace , bbox = aligner.prepocessImg('affine', 128, id_img, bb, offset=0.3)
#fea = extractor.extract_batch(face)
fea = extractor.extract_once(alignedFace)
#while(True):
#    # Capture frame-by-frame
#    ret, img = cap.read()
#    try:
#        bbox = aligner.getAllFaceBoundingBoxes(img)
#        img = aligner.drawboxs(img,bbox)
#    except:
#        pass
#    # Display the resulting frame
#    cv2.imshow('image',img)
#    alignedFace , bbox = aligner.prepocessImg('affine', 128, img, bbox, offset=0.3)
#    fea = extractor.extract_batch(alignedFace)
#    print 'feature get'
#
#    
##    if cv2.waitKey(1) & 0xFF == ord('m'):
#       
#        
#    
## When everything done, release the capture
#cap.release()
#cv2.destroyAllWindows()
