"""
this script is use landmark to do similiar transform for detected face
"""
from PIL import Image
import math
from coodinate_convert import xy_to_wh


def Distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx*dx+dy*dy)


def ScaleRotateTranslate(image, angle, center=None, new_center=None,
                         scale=None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    new_x, new_y = x, y = center
    scale_x = scale_y = 1.0
    if new_center:
        (new_x, new_y) = new_center
    if scale:
        (scale_x, scale_y) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine/scale_x
    b = sine/scale_x
    c = x-new_x*a-new_y*b
    d = -sine/scale_y
    e = cosine/scale_y
    f = y-new_x*d-new_y*e
    return image.transform(image.size, Image.AFFINE,
                           (a, b, c, d, e, f), resample=resample)


def crop_simi(image, eye_left=(0, 0), eye_right=(0, 0),
              offset_pct=(0.2, 0.2), dest_sz=(70, 70)):
    """
    this function is used to do similarity transformation,
    put the eye into horizontal direction
    and make the eye in the same position
    """
    # calculate offsets in output image
    offset_h = math.floor(float(offset_pct[0])*dest_sz[0])  # h means hotizontal
    offset_v = math.floor(float(offset_pct[1])*dest_sz[1])  # v means vertical
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians,the horizental anggle
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # distance between them
    dist = Distance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0*offset_h
    # scale factor
    # 1/scale is how the final image change
    scale = float(dist)/float(reference)  # the length between two eye is fixed
    # rotate original around the left eye
    # just rotate the image without scale
    image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
    # crop the rotated image
    # the coodinate of left eye is fixed
    crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
    crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
    image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]),
                        int(crop_xy[1]+crop_size[1])))
    # resize it
    image = image.resize(dest_sz, Image.ANTIALIAS)
    return image


def crop_only(image, xy, extend=0.1, new_dim=224):
    """
    this function is used to crop face into  square shape
    the order of coordinate truplr should be (leftx,rightx,upy,downy)
    """
    left_x, up_y, width, height = xy_to_wh(xy)
    length = (width+height)/2
    center_x = left_x + width/2
    center_y = up_y + height/2
    im_width, im_height = image.size
    left_x = int(max(center_x - round((1+extend)*length/2), 0))
    up_y = int(max(center_y - round((1+extend)*length/2), 0))
    right_x = int(min(center_x + round((1+extend)*length/2), im_width))
    down_y = int(min(center_y + round((1+extend)*length/2), im_height))

    image = image.crop((left_x, up_y, right_x, down_y))
    image = image.resize((new_dim, new_dim))
    return image
