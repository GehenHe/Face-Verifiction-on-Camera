#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:45:13 2017

@author: gehen
"""

import numpy as np
import cv2
from preprocess import NaiveDlib,Wrapper,evaluator
from PIL import Image
from time import time

aligne_model = './model/dlib/shape_predictor_68_face_landmarks.dat'
caffe_model = './model/caffe/deepface/deepface.caffemodel'
caffe_proto = './model/caffe/deepface/val.prototxt'
mean_file = './model/caffe/deepface/mean.npy'
id_img = cv2.imread('./data/Amanda_Bynes/Amanda_Bynes_0001.jpg')
aligner = NaiveDlib(aligne_model)
extractor = Wrapper(caffe_proto, caffe_model, mean_file)
eva = evaluator('cos','accuracy')
thresh = 0.7

cap = cv2.VideoCapture(-1)
bb = aligner.getLargestFaceBoundingBox(id_img)
alignedFace , bbox = aligner.prepocessImg('affine', 128, id_img, bb, offset=0.3)
ID_fea = extractor.extract_once(alignedFace)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    try:
        bbox = aligner.getLargestFaceBoundingBox(img)
        img = aligner.drawboxs(img,bbox)
    except:
        pass
    # Display the resulting frame
    cv2.imshow('image',img)
    key = cv2.waitKey(1)
    if key:
        if key&0xFF == ord('v'):
            try:
                start = time()
                alignedFace , bbox = aligner.prepocessImg('affine', 128, img, bbox, offset=0.3)
                fea = extractor.extract_once(alignedFace)
                eva.cal_score(ID_fea,fea)
                score = eva.score[0]
                end = time()
                print "took {:.3f} s ".format(end-start)
                if score>thresh:
                    print 'Same person and score is {:.3f}\n'.format(score)
                else:
                    print 'Different person and score is {:.3f}\n'.format(score)
            except:
                print "no face is detected"
        if key&0xFF == ord('q'):
            break
        else:
            pass 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()