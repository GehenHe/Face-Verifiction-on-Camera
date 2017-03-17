# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 15:27:17 2016

@author: teddy
"""


def xy_to_wh(xy):
    """
    input is (leftx,rightx,upy,downy)
    output is (leftx,upy,width,height)
    """

    return (xy[0], xy[2], xy[1]-xy[0], xy[3]-xy[2])


def wh_to_xy(xy):
    """
    input is (leftx,upy,width,height)
    output is (leftx,rightx,upy,downy)
    """

    return (xy[0], xy[0]+xy[2], xy[1], xy[1]+xy[3])
