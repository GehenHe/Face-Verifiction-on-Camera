# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 20:31:32 2015
Extrator is a wrapper of Net which used to extract feature from images 
@author: teddy
"""

import numpy as np

import caffe

class Extractor(caffe.Net):
    """
    Extrator extends Net for feature extrating
    by scaling, center cropping
    Parameters
    ----------
    image_dims : dimensions to scale input for cropping.
        Default is to scale to net input size for whole-image crop.
    mean, input_scale, raw_scale, channel_swap: params for
        preprocessing options.
    """
    def __init__(self, model_file, pretrained_file, image_dims=None,
                 mean=None, input_scale=None, raw_scale=None,
                 channel_swap=None):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        # configure pre-processing
        in_ = self.inputs[0]                               #the first key of ordered dic,name str
        self.transformer = caffe.io.Transformer(
            {in_: self.blobs[in_].data.shape})             #transformer's initial data_format
        if len(self.blobs[in_].data.shape) >= 3:
            self.transformer.set_transpose(in_, (2, 0, 1))     #channel is in front od h and w
        if mean is not None:
            self.transformer.set_mean(in_, mean)
        if input_scale is not None:                        #multiply after mean
            self.transformer.set_input_scale(in_, input_scale)
        if raw_scale is not None:                          #multiply before mean 255 (0,1) transfer
            self.transformer.set_raw_scale(in_, raw_scale)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:]) #make sure it can be input into the net
        #crop_dim is the shape of blob
        #image_dim is the shape of image
        if not image_dims:                               #im_idm is used to resize photo
            image_dims = self.crop_dims
        self.image_dims = image_dims                    #im_dim should be defined if you want to do sample

    def extract_batch(self, inputs,blob = None,oversample= False):
        """
        extract image feature in bach  
        Parameters
        ----------
        inputs : iterable of (H x W x K) input ndarrays.
        inputs can be a list of (H*W*K) or can be a N*H*W*K ndarray
        blobs_name: the name of blobs which you want to save

        Returns 
        -------
        name_feature dic which is defined by blobs_name     
        """
        # Scale to standardize input dimensions.
        input_ = np.zeros((len(inputs),             #the num of input image
                           self.image_dims[0],
                           self.image_dims[1],
                           inputs[0].shape[2]),    #the channel
                          dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            #if input image is crop dim do no transform
            input_[ix] = caffe.io.resize_image(in_, self.image_dims)  #this function
                                                                      #may introduce interpolation           
        if oversample:
            # Generate center, corner, and mirrored crops.
            input_ = caffe.io.oversample(input_, self.crop_dims)
        else:
            # Take center crop.
            center = np.array(self.image_dims) / 2.0
            crop = np.tile(center, (1, 2))[0] + np.concatenate([
                -self.crop_dims / 2.0,
                self.crop_dims / 2.0
            ])
            input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]
        # extact feature
        caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                            dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
        if not blob: 
            out = self.forward_all(**{self.inputs[0]: caffe_in})[self.outputs[0]]
        else:
            out = self.forward(blobs = [blob],**{self.inputs[0]: caffe_in})[blob]
            
        # For oversampling, average predictions across crops.
        if oversample:
            out = out.reshape((len(out) / 10, 10, -1))
            out = out.mean(1)
        return out
    def extract_only(self, inputs,blob = None,norm = True,batch_size = 100):
        in_ = self.inputs[0]
        if norm:
            norm_fea = np.linalg.norm(inputs,axis = 1)
            norm_fea = norm_fea.reshape(norm_fea.shape[0],1)
            inputs = inputs/norm_fea
            del norm_fea
        length =  inputs.shape[0]
        loop = length/batch_size
        remain = length%batch_size
        print('the total num of batch is {0:d}'.format(loop+int(remain>0)))
        out_fea = []
        for i in range(loop):
            if not blob: 
                out = self.forward_all(**{in_: inputs[i*batch_size:(i+1)*batch_size]})[self.outputs[0]]
                out_fea.append(out)
            else:
                out = self.forward([blob],**{in_: inputs[i*batch_size:(i+1)*batch_size]})[blob]
                out_fea.append(out)
        if remain != 0:
            print('extract batch {0:d}'.format(loop))
            if not blob: 
                out = self.forward_all(**{in_: inputs[batch_size*loop:]})[self.outputs[0]]
                out_fea.append(out)
            else:
                out = self.forward([blob],**{in_: inputs[batch_size*loop:]})[blob]
                out_fea.append(out)
        out_fea = np.vstack(out_fea)
        return out_fea
  
    def extract_list(self,img_list,blob = None,oversample = False,color= True,batch_size = 100):
        length =  len(img_list)
        loop = length/batch_size
        remain = length%batch_size
        fea = []
        print('the total num of batch is {0:d}'.format(loop+int(remain>0)))
        for i in range(loop):
            print('extract batch {0:d}'.format(i+1))
            inputs = []
            for img in img_list[i*batch_size:(i+1)*batch_size]:
                inputs.append(caffe.io.load_image(img,color = color))
            features = self.extract_batch(inputs,blob = blob,oversample = oversample)
            fea.append(features)
        if remain != 0:
            print('extract batch {0:d}'.format(loop))
            inputs = []
            for img in img_list[loop*batch_size:]:
                inputs.append(caffe.io.load_image(img,color = color))
            features = self.extract_batch(inputs,blob = blob,oversample = oversample)
            fea.append(features)
        fea = np.vstack(fea)
        return fea
        
