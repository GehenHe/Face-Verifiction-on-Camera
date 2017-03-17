# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:27:28 2015
this module is used for visualization
it's a package of pil module
and all the method is based on pil module
@author: teddy
"""
from PIL import ImageDraw, ImageFont


class Color:
    lookup_tabel = {
      'white': (0, 0, 0),
      'red': (255, 0, 0),
      'green': (0, 255, 0),
      'blue': (0, 0, 255),
      'cyan': (0, 255, 255),
      'magenta': (255, 0, 255),
      'yellow ': (212, 212, 0),
      'black ': (25, 25, 25),
      'forestgreen': (34, 139, 34),
      'deepskyblue': (0, 191, 255),
      'darkred': (139, 0, 0),
      'orchid': (218, 112, 214),
      'sandybrown': (244, 164, 96)
    }

    def __init__(self, color_index):
        self.rgb = None
        if isinstance(color_index, int):
            self.getcolorfromid(color_index)
        elif isinstance(color_index, str):
            self.getcolorfromname(color_index)
        else:
            raise ValueError("index should be an int type or a string type")
            self.rgb = Color.lookup_tabel['red']

    def getcolorfromname(self, name):
        try:
            self.rgb = Color.lookup_tabel[name]
        except KeyError:
            print "Error: can't find the name in color lookup_table"

    def getcolorfromid(self, id_num):
        all_num = len(Color.lookup_tabel)
        idx = id_num % all_num
        self.rgb = list(Color.lookup_tabel.values())[idx]


def drawbox(im, xy, color_index='red', width=5):
    """
    the color index can be string or num
    you can set the color and the width of bbox
    the order of coordinate truplr should be (leftx,rightx,upy,downy)
    """

    color = Color(color_index)
    draw = ImageDraw.Draw(im)
    up_line = [xy[0], xy[2], xy[1], xy[2]]
    down_line = [xy[0], xy[3], xy[1], xy[3]]
    left_line = [xy[0], xy[2], xy[0], xy[3]]
    right_line = [xy[1], xy[2], xy[1], xy[3]]
    draw.line(up_line, color.rgb, width)
    draw.line(down_line, color.rgb, width)
    draw.line(left_line, color.rgb, width)
    draw.line(right_line, color.rgb, width)
    del draw


def drawtext(im, point, my_str, color_index='red',
             font='ukai.ttc', size=60):
    """
    coodinate is the left_up point of the string
    point is a truple
    """
    color = Color(color_index)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(font, size)
    draw.text(point, my_str, color, font=font)
    del draw


def drawpoint(im, center_point, color_index='green', radius=5):
    """
    the size is point's radius,xy is the center of of the circle
    """
    color = Color(color_index)
    draw = ImageDraw.Draw(im)
    draw.pieslice((center_point[0]-radius, center_point[1]-radius,
                   center_point[0]+radius, center_point[1]+radius),
                  0, 360, color.rgb)
    del draw
