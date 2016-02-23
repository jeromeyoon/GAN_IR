"""
Some codes from https://github.com/Newmu/dcgan_code
"""
import math
import json
import pprint
import scipy.misc
import numpy as np

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_image(image_path, image_size,randx,randy, is_crop=True):
    return transform(imread(image_path), image_size, randx,randy,is_crop)

def get_image_normal(image_path, image_size,randx,randy, is_crop=True):
    return transform_normal(normalize(imread(image_path)), image_size, randx,randy,is_crop)
#def get_image(image_path, image_size, is_crop=True):
#    return transform(imread(image_path), image_size, is_crop)

def get_image_eval(image_path):
    return transform_eval(imread(image_path))

def save_images(images, size, image_path):
    return imsave(inverse_transform(inverse_normalize(images)), size, image_path)
#def save_images(images, size, image_path):
#    return imsave(inverse_transform(images), size, image_path)

def imread(path):
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def normalize(images):
    images = images/255.0 # 0~1
    images = (images * 2.0) -1.0 # -1~1
    z = images[:, :, -1]
    z.clip(np.exp(-100), z.max(),out=z)
    isZero = np.where(z == np.exp(-100))
    yy = images[:,:,0]/z
    xx = images[:,:,1]/z

    yy[isZero] = 0.0
    xx[isZero] = 0.0

    tmp = np.dstack((yy,xx))
    overone = np.where(tmp >1.0)
    lessone = np.where(tmp < -1.0)
    tmp[overone] = 1.0
    tmp[lessone] = -1.0
    return tmp



def inverse_normalize(images):

    y = images[:,:,:,0]
    x = images[:,:,:,1]
    z = np.full((images.shape[0],images.shape[1],images.shape[2]),1,dtype=np.float)
    is_zero = np.where(x == 0)
    norm = np.sqrt(np.power(x,2)+np.power(y,2)+1.)
    yy = y/norm
    xx = x/norm
    zz = z/norm

    yy[is_zero]= 0.0
    xx[is_zero]= 0.0
    zz[is_zero]= 0.0

    inv = np.stack((yy,xx,zz),axis=-1)

    #print('inv shape:',inv.shape)
    #inv = np.stack(np.stack((yy,xx,zz),axis=0))
    return inv


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def random_crop(x,npx,randx,randy):
    npx =64
    return x[randy:randy+npx, randx:randx+npx]

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])
"""
def transform(image, npx=64, is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.
"""
def transform_normal(image, npx, randx,randy,is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = random_crop(image, npx,randx,randy)
        #cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    #scipy.misc.imshow(cropped_image)
    #print('cropped image dim:',cropped_image.shape)
    #print('x:%d y:%d' % (randx,randy))
    return np.array(cropped_image)/0.5 - 1.


def transform(image, npx, randx,randy,is_crop=True):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = random_crop(image, npx,randx,randy)
        #cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    #scipy.misc.imshow(cropped_image)
    #print('cropped image dim:',cropped_image.shape)
    #print('x:%d y:%d' % (randx,randy))
    return np.array(cropped_image)/127.5 - 1.


def inverse_transform(images):
    return (images+1.)/2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
                    var layer_%s = {
                        "layer_type": "fc",
                        "sy": 1, "sx": 1,
                        "out_sx": 1, "out_sy": 1,
                        "stride": 1, "pad": 0,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
                    var layer_%s = {
                        "layer_type": "deconv",
                        "sy": 5, "sx": 5,
                        "out_sx": %s, "out_sy": %s,
                        "stride": 2, "pad": 1,
                        "out_depth": %s, "in_depth": %s,
                        "biases": %s,
                        "gamma": %s,
                        "beta": %s,
                        "filters": %s
                    };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
                             W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)