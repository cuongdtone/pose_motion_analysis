from PIL import Image
import os
import cv2
import numpy as np
import random
import glob
import mxnet as mx

def get_interp(interp, sizes=()):
    """Get the interpolation method for resize functions.
    The major purpose of this function is to wrap a random interp method selection
    and a auto-estimation method.

    Parameters
    ----------
    interp : int
        interpolation method for all resizing operations

        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Bicubic interpolation over 4x4 pixel neighborhood.
        3: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
        More details can be found in the documentation of OpenCV, please refer to
        http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.
    sizes : tuple of int
        (old_height, old_width, new_height, new_width), if None provided, auto(9)
        will return Area(2) anyway.

    Returns
    -------
    int
        interp method from 0 to 4
    """
    if interp == 9:
        if sizes:
            assert len(sizes) == 4
            oh, ow, nh, nw = sizes
            if nh > oh and nw > ow:
                return 2
            elif nh < oh and nw < ow:
                return 3
            else:
                return 1
        else:
            return 2
    if interp == 10:
        return random.randint(0, 4)
    if interp not in (0, 1, 2, 3, 4):
        raise ValueError('Unknown interp method %d' % interp)
    return interp


def resize_short_within(src, short, max_size, mult_base=1, interp=2):
    """Resizes shorter edge to size but make sure it's capped at maximum size.
    Note: `resize_short_within` uses OpenCV (not the CV2 Python library).
    MXNet must have been built with OpenCV for `resize_short_within` to work.
    Resizes the original image by setting the shorter edge to size
    and setting the longer edge accordingly. Also this function will ensure
    the new image will not exceed ``max_size`` even at the longer side.
    Resizing function is called from OpenCV.

    Parameters
    ----------
    src : NDArray
        The original image.
    short : int
        Resize shorter side to ``short``.
    max_size : int
        Make sure the longer side of new image is smaller than ``max_size``.
    mult_base : int, default is 1
        Width and height are rounded to multiples of `mult_base`.
    interp : int, optional, default=2
        Interpolation method used for resizing the image.
        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        3: Bicubic interpolation over 4x4 pixel neighborhood.
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method mentioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
        More details can be found in the documentation of OpenCV, please refer to
        http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.
    Returns
    -------
    NDArray
        An 'NDArray' containing the resized image.
    Example
    -------
    >>> with open("flower.jpeg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.img.imdecode(str_image)
    >>> image
    <NDArray 2321x3482x3 @cpu(0)>
    >>> new_image = resize_short_within(image, short=800, max_size=1000)
    >>> new_image
    <NDArray 667x1000x3 @cpu(0)>
    >>> new_image = resize_short_within(image, short=800, max_size=1200)
    >>> new_image
    <NDArray 800x1200x3 @cpu(0)>
    >>> new_image = resize_short_within(image, short=800, max_size=1200, mult_base=32)
    >>> new_image
    <NDArray 800x1184x3 @cpu(0)>
    """
    h, w, _ = src.shape
    im_size_min, im_size_max = (h, w) if w > h else (w, h)
    scale = float(short) / float(im_size_min)
    if np.round(scale * im_size_max / mult_base) * mult_base > max_size:
        # fit in max_size
        scale = float(np.floor(max_size / mult_base) * mult_base) / float(im_size_max)
    new_w, new_h = (int(np.round(w * scale / mult_base) * mult_base),
                    int(np.round(h * scale / mult_base) * mult_base))
    return imresize(src, new_w, new_h, interp=get_interp(interp, (h, w, new_h, new_w)))


def imresize(src, w, h, interp=1):
    """Resize image with OpenCV.

    This is a duplicate of mxnet.image.imresize for name space consistency.

    Parameters
    ----------
    src : mxnet.nd.NDArray
        source image
    w : int, required
        Width of resized image.
    h : int, required
        Height of resized image.
    interp : int, optional, default='1'
        Interpolation method (default=cv2.INTER_LINEAR).

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.

    Examples
    --------
    >>> import mxnet as mx
    >>> from gluoncv import data as gdata
    >>> img = mx.random.uniform(0, 255, (300, 300, 3)).astype('uint8')
    >>> print(img.shape)
    (300, 300, 3)
    >>> img = gdata.transforms.image.imresize(img, 200, 200)
    >>> print(img.shape)
    (200, 200, 3)
    """
#    from mxnet.image.image import _get_interp_method as get_interp
    oh, ow, _ = src.shape
    return mx.image.imresize(src, w, h, interp=get_interp(interp, (oh, ow, h, w)))
    

def read_an_image_PIL(image_path='./sample.jpg'):
    im = Image.open(image_path)
    return im

def read_an_image_opencv(image_path='./sample.jpg'):
    im = cv2.imread(image_path)
    return im

def write_an_image_PIL(image, image_path='./sample.jpg'):
    image.save(image_path)

def write_an_image_opencv(image, image_path='./sample.jpg'):
    cv2.imwrite(image, image_path)

def resize_and_pad_image_to_square_opencv(image, desired_size=224):
    old_size = image.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    image = cv2.resize(image, (new_size[1], new_size[0])) 

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_image


def resize_and_pad_image_to_square_PIL(image, desired_size=224):
    old_size = image.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    resized_img = image.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_img = Image.new("RGB", (desired_size, desired_size))
    new_img.paste(resized_img, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
    return new_img

def resize_and_pad_image_dir(image_dir, out_dir, ext='*.tif', desired_size=224):
    image_paths = sorted(glob.glob(os.path.join(image_dir,ext)))
    for image_path in image_paths:
        base_name = os.path.basename(image_path)
        out_path = os.path.join(out_dir,base_name)
        img = read_an_image_opencv(image_path)
        new_img = resize_and_pad_image_to_square_opencv(img)
        write_an_image_opencv(out_path, new_img)

def resize_and_pad_all_image_dirs(src_dir, dst_dir, ext='*.tif', desired_size=224):
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    image_dirs = [f.path for f in os.scandir(src_dir) if f.is_dir()]
    for i, image_dir in enumerate(image_dirs):
        print('Processing the clip {}/{}-th: {}'.format(i,len(image_dirs),image_dir))
        base_name = os.path.basename(image_dir)
        out_dir = os.path.join(dst_dir,base_name)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            resize_and_pad_image_dir(image_dir, out_dir)

def clip_dir_to_video(clip_dir, video_path, ext='*.tif'):
    image_paths = sorted(glob.glob(os.path.join(clip_dir, ext)))
    if len(image_paths)>0:
        image_path = image_paths[0]
        frame = cv2.imread(image_path)
        height = frame.shape[0]
        width = frame.shape[1]

        # Set up parameters for VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 20, (width, height), True)

        for image_path in image_paths:
            frame = cv2.imread(image_path)        
            out.write(frame)
        
        out.release()

def clips_dir_to_videos(clips_dir, ext='*.tif', out_dir='./outputs/'):
    clip_dirs = [f.path for f in os.scandir(clips_dir) if f.is_dir()]
    for i, clip_dir in enumerate(clip_dirs):
        print('Processing the clip {}/{}-th: {}'.format(i,len(clip_dirs),clip_dir))
        video_name = os.path.basename(clip_dir)
        video_path = os.path.join(out_dir, video_name + '.mp4')
        if not os.path.exists(video_path):
            clip_dir_to_video(clip_dir, video_path, ext)

def test3():
    clips_dir = '/media/hueailab/HDPC-UT/Documents/DeTaiDHH/Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1_resized/Test'
    out_dir = '/media/hueailab/HDPC-UT/Documents/DeTaiDHH/Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1_resized/Test_mp4'
    ext = '*.tif'
    clips_dir_to_videos(clips_dir, ext, out_dir)

def test1():
    im_pth = '/media/hueailab/HDPC-UT/Documents/DeTaiDHH/Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001/001.tif'
    img = read_an_image(im_pth)
    new_img = resize_and_pad_image_to_square(img)
    new_img.show()
    
def test2():
    ### UCSDped2
    src_dir = '/media/hueailab/HDPC-UT/Documents/DeTaiDHH/Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Train'
    dst_dir = '/media/hueailab/HDPC-UT/Documents/DeTaiDHH/Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2_resized/Train'
    ext = '*.tif'
    desired_size = 224
    resize_and_pad_all_image_dirs(src_dir, dst_dir, ext, desired_size)
    
#    src_dir = '/media/hueailab/HDPC-UT/Documents/DeTaiDHH/Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test'
#    dst_dir = '/media/hueailab/HDPC-UT/Documents/DeTaiDHH/Dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2_resized/Test'
#    ext = '*.tif'
#    desired_size = 224
#    resize_and_pad_all_image_dirs(src_dir, dst_dir, ext, desired_size)


if __name__=='__main__':
#    test2()
    test3()
