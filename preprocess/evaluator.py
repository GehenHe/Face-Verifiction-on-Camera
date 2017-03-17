# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:40:28 2015
the evaluation protocal
@author: teddy
"""
import measure
import protocol

class evaluator(object):
    def __init__(self,simi_mea,eval_pro):
        self.__measure = simi_mea
        self.__protocol = eval_pro
    def set_measure(self,simi_mea):
        self.__measure = simi_mea
    def set_protocol(self,eval_pro):
        self.__protocol = eval_pro
    def cal_score(self,fea1,fea2):
        if self.__measure == 'L2':
            self.score = measure.compute_dis(fea1,fea2)
        elif self.__measure == 'cos':
            self.score = measure.cos(fea1,fea2)
    def evaluate(self,label,thre = None):
        if self.__protocol == 'accuracy':
            return protocol.best_acc(self.score,label)   
        elif self.__protocol == 'thre':
            return protocol.thre_acc(self.score,label,thre)
        elif self.__protocol == 'roc':
            return protocol.roc(self.score,label)
        elif self.__protocol == 'wrong':
            return protocol.find_wrong(self.score,label,thre)
