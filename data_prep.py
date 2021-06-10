from PIL import Image
import numpy as np
import cv2
from os import listdir
# from os.path import isfile, join


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    if h % nrows != 0:
        return 'pass'
    if w % ncols != 0:
        return 'pass'
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)  # noqa
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)  # noqa
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))


def downPixelAggr(img, SCALE=1):
    # from scipy import signal
    import skimage.measure
    from scipy.ndimage.filters import gaussian_filter

    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_blur = np.zeros(img.shape)
    # Filter the image with a Gaussian filter
    for i in range(0, img.shape[2]):
        img_blur[:, :, i] = gaussian_filter(img[:, :, i], 1/SCALE)
    # New image dims
    new_dims = tuple(s//SCALE for s in img.shape)
    img_lr = np.zeros(new_dims[0:2]+(img.shape[-1],))
    # Iterate through all the image channels with avg pooling (pixel aggregation)  # noqa
    for i in range(0, img.shape[2]):
        img_lr[:, :, i] = skimage.measure.block_reduce(img_blur[:, :, i], (SCALE, SCALE), np.mean)  # noqa

    return np.squeeze(img_lr)


def tuple_insert(tup, pos, ele):
    tup = tup[:pos]+(ele,)+tup[pos:]
    return tup


def lr_hr():
    print('degisti')
    mypath = '/content/drive/My Drive/IP4RS/data/UCMerced_LandUse/Images'

    # sub image size
    img_size_hr = 64
    img_size_lr = img_size_hr

    # defining arrays
    arr_hr = np.ndarray(shape=(0, img_size_hr, img_size_hr))
    arr_lr = np.ndarray(shape=(0, img_size_lr, img_size_lr))

    folders = listdir(mypath)

    # iteration over files
    for folder in folders:
        # print(folder)
        only_files = listdir(mypath+'/'+folder+'/')
        i = 0
        count = 0
        # reading every first 50 items in each folder
        while i < 50:

            try:
                i += 1
                # reading the data
                im = Image.open(mypath+'/'+folder+'/'+only_files[i])

                # converting to np array
                image_np = np.array(im)

                # converting to grayscale
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                # normalizing
                gray = gray/256

                shape_ind = int(gray.shape[0]/img_size_hr)**2

                # creating sub images from original images
                gray_hr = blockshaped(gray,
                                      img_size_hr,
                                      img_size_hr)

                # creating empty array to fill it with low resolution sub images  # noqa
                empty_lr = np.ndarray(shape=(shape_ind,
                                             img_size_lr,
                                             img_size_lr))

                # poulation empty array with down sampled images
                for ind in range(len(gray_hr)):
                    empty_lr[ind] = downPixelAggr(gray_hr[ind])

                # populating final high resolution array
                arr_hr = np.concatenate((arr_hr, gray_hr))
                # print(arr_hr.shape)

                # populating final low resolution array
                arr_lr = np.concatenate((arr_lr, empty_lr))
            except:  # noqa
                count += 1
    print('Skipped: '+str(count))
    new_shape_hr = tuple_insert(arr_hr.shape, 1, 1)
    new_shape_lr = tuple_insert(arr_lr.shape, 1, 1)

    # reshaping
    arr_dhr = arr_hr.reshape(new_shape_hr)
    arr_dlr = arr_lr.reshape(new_shape_lr)

    return arr_dhr, arr_dlr
