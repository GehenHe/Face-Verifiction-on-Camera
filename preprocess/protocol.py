# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:26:19 2015

@author: teddy
"""
import numpy as np
import matplotlib.pyplot as plt

def best_acc(score_arr,label_arr):
    thre = float("-inf")
    count = 0
    for temp_thre in score_arr:
        temp_count = 0
        for idx,score in enumerate(score_arr):
            if score >= temp_thre:
                pre_label = 1
            else:
                pre_label = 0
            if pre_label == label_arr[idx]:
                temp_count+=1
        if temp_count > count:
            count = temp_count
            thre = temp_thre
    acc = 1.0*count/score_arr.shape[0]
    return acc,thre
    
def thre_acc(score_arr,label_arr,thre):
    count = 0
    for idx,score in enumerate(score_arr):
            if score >= thre:
                pre_label = 1
            else:
                pre_label = 0
            if pre_label == label_arr[idx]:
                count+=1
    acc_score = 1.0*count/score_arr.shape[0]
    return acc_score

def roc(score_arr,label_arr):
    thre_arr = np.sort(score_arr)
    fpr_list = []
    tpr_list = []
    p_count = 0
    n_count = 0
    delta = 1
    eer = 0
    for label in label_arr:
        if label == 1:
            p_count+=1
        elif label == 0:
            n_count+=1        
    for thre in thre_arr:
        tp_count = 0
        fp_count = 0
        for idx,score in enumerate(score_arr):
            if score>=thre:
                pre_label = 1
            else:
                pre_label = 0
            if label_arr[idx] == 1 and pre_label == 1:
                tp_count+=1
            elif label_arr[idx] == 0 and pre_label == 1:
                fp_count+=1
        tpr = 1.0*tp_count/p_count
        fpr = 1.0*fp_count/n_count
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        temp_delta = abs(tpr+fpr-1)
        if temp_delta<delta:
            delta = temp_delta
            eer =tpr
    tpr_list.append(0)
    fpr_list.append(0)
    plt.plot(fpr_list,tpr_list,linewidth=1.0)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.show()
    return fpr_list,tpr_list,eer

def find_wrong(score_arr,label_arr,thre):
    fn =[]
    fp =[]
    for idx,score in enumerate(score_arr):
        if score >= thre:
            pre_label = 1
        else:
            pre_label = 0
        if pre_label != label_arr[idx]:
            if label_arr[idx] == 1:
                fn.append(idx)
            else:
                fp.append(idx)
    return fn,fp
                
    
    
if __name__ == '__main__':
    from measure import compute_dis,cos
    acc_list = []
    thre_list = []
    for num in range(10):
        fea1 = np.load('/media/teddy/data/lfw_work_dir/fea_vgg/test_fea_l_{0:d}.npy'.format(num))
        fea2 = np.load('/media/teddy/data/lfw_work_dir/fea_vgg/test_fea_r_{0:d}.npy'.format(num))
        label = np.load('/media/teddy/data/lfw_work_dir/fea_vgg/test_label_{0:d}.npy'.format(num))
        score = compute_dis(fea1,fea2)
        acc,thre = best_acc(score,label)
        acc_list.append(acc)
        thre_list.append(thre)
    mean = sum(acc_list)/len(acc_list)
        
        
        