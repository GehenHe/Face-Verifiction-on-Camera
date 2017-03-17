import dlib
import numpy as np
import os,cv2
from canonical_face import crop_simi,crop_only
from visualization import drawbox,drawpoint
import base64
import cStringIO
from PIL import Image


class PreError(Exception):
    pass

class DetError(PreError):
    pass

class AliError(PreError):
    pass



class NaiveDlib:

    def __init__(self, facePredictor = None):
        """Initialize the dlib-based alignment."""
        self.detector = dlib.get_frontal_face_detector()
        if facePredictor != None:
            self.predictor = dlib.shape_predictor(facePredictor)
        else:
            self.predictor = None 

    def getAllFaceBoundingBoxes(self, img):
        faces = self.detector(np.array(img), 1)
        if len(faces)>0:
            return faces
        else:
            raise DetError('cannot detect any face')

    def getLargestFaceBoundingBox(self, img):    #process only one face pertime
        faces = self.detector(np.array(img), 1)
        if len(faces) > 0:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            raise DetError('cannot detect any face')

    def align(self, img, bb):
        points = self.predictor(np.array(img), bb)
        return list(map(lambda p: (p.x, p.y), points.parts()))
                
    def prepocessImg(self, method, size, img, bb,offset=0.3,gray=True,
                     boundry=False, outputDebug=False,outputprefix=None):
        """
        the image is load by PIL image directly
        bb is rct object of dlib
        """
        if method == 'crop':
            crop_img = crop_only(img,bb.left(),bb.top(),bb.width(),bb.height(),offset,size)
        elif method == 'affine':
            img = Image.fromarray(img)
            if self.predictor == None:
                raise Exception("Error: method affine should initial with an facepredictor.")
            alignPoints = self.align(img, bb)
            (xs, ys) = zip(*alignPoints)
            (l, r, t, b) = (min(xs), max(xs), min(ys), max(ys))
            w,h = img.size
            if boundry and (l < 0 or r > w or t < 0 or b > h):
                raise AliError('face out of boundry')
                
            left_eye_l = alignPoints[36]
            left_eye_r = alignPoints[39]
            left_eye = (np.array(left_eye_l)+np.array(left_eye_r))/2
            right_eye_l = alignPoints[42]
            right_eye_r = alignPoints[45]
            right_eye = (np.array(right_eye_l)+np.array(right_eye_r))/2
            crop_img = crop_simi(img,left_eye,right_eye,(offset,offset),(size,size))
            im_buffer = cStringIO.StringIO()
            crop_img.save(im_buffer, format="JPEG")
            im_str = base64.b64encode(im_buffer.getvalue())
        else:
            raise Exception("undefined crop method")
        if gray:
            crop_img = crop_img.convert('L')
        if outputDebug:
            dirname = './aligndebug'
            if not os.path.exists(os.path.abspath(dirname)):
                os.mkdir(dirname)
            drawbox(img,(bb.left(),bb.right(),bb.top(),bb.bottom()))
            if method == 'affine':
                drawpoint(img,left_eye)
                drawpoint(img,right_eye)
            img.save('{}/{}_annotated.jpg'.format(dirname,outputprefix))
            crop_img.save('{}/{}_crop.jpg'.format(dirname,outputprefix))
        crop_img = np.array(crop_img,dtype=np.float32)      #look carefully on data format
        if crop_img.ndim == 3:                              #data shape for caffe
            return crop_img,score
        elif crop_img.ndim == 2:
            bbox = [bb.left(),bb.top(),bb.right(),bb.bottom()]
            return crop_img[:,:,np.newaxis], bbox
        else:
            raise Exception("wrong dimension")

    def drawboxs(sefl,img,bb):
        cv2.rectangle(img,(bb.left(),bb.top()),(bb.right(),bb.bottom()),(0,255,0),2)
        return img


if __name__ == '__main__':
    from PIL import Image
    aligne_model = '../model/dlib/shape_predictor_68_face_landmarks.dat'
    im_dir = '/home/gehen/facesearch/data/pic/10001488699533828117.jpg'
    aligner = NaiveDlib(aligne_model)
    img =  Image.open(im_dir)
    bb = aligner.getLargestFaceBoundingBox(img)
    alignedFace,score =  aligner.prepocessImg('affine', 128, img, bb,offset = 0.3,boundry=True,gray=True,
                                         outputDebug=True,outputprefix = '4'
                                       )
    
