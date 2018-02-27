import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import shift
from scipy.ndimage.morphology import binary_erosion

from skimage.feature import register_translation
from skimage.filters import sobel, threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import label
from skimage.io import imsave

import keras
upsampler = keras.models.load_model('upsample_model.h5')

def calculate_shift(imgA, imgB, pxwidth):
    """Uses the registration algorithm from scikit-image
    to calculate alignment to within 0.01 pixels or 1 nm,
    whichever is coarser."""
    
    sigma = 0.15/pxwidth
    filtA = gaussian_filter(imgA, min(sigma, 3.0))
    filtB = gaussian_filter(imgB, min(sigma, 3.0))
    
    thr = threshold_otsu(filtB)
    edgecheck = 1*(filtB < thr)
    # Check whether the object lies along a boundary of the image,
    # and if so apply a Sobel filter to use edges to align
    (a, b) = edgecheck.shape
    edgects = [np.sum(edgecheck[0])/float(b), np.sum(edgecheck[-1])/float(b), np.sum(edgecheck[:,0])/float(a), np.sum(edgecheck[:,-1])/float(a)]
    if max(edgects) > 0.4:
        filtA = sobel(filtA)
        filtB = sobel(filtB)
    
    shift = register_translation(filtA, filtB, upsample_factor = min(1000.0, 100.0/pxwidth))
    return [-shift[0][1], shift[0][0]]

def alignoneimage(img, sh):
    """Given an image and x,y shifts in pixels, returns a
    shifted image in which pixels beyond the boundaries of
    the original image have their values set to -1.
    """
    aligned = shift(img, [sh[1], -sh[0]], mode = 'nearest')
    
    (u, v) = img.shape
    
    if sh[1] >= 0.5:
        irange = range(int(sh[1])+1)
    elif sh[1] <= -0.5:
        irange = range(u - int(-sh[1]), u)
    else:
        irange = []
    for i in irange:
        for j in range(v):
            aligned[i][j] = -1
    
    if sh[0] >= 0.5:
        jrange = range(v - int(sh[0]), v)
    elif sh[0] <= -0.5:
        jrange = range(int(-sh[0])+1)
    else:
        jrange = []
    for i in range(u):
        for j in jrange:
            aligned[i][j] = -1
        
    return aligned

def alignstack(raw, pxwidth, master):
    """Given a stack of images and the width of each pixel (in
    microns), calculates the shifts needed to align all images
    with the highest-contrast image in the stack.
    
    Returns the list of shifts and a stack of aligned images.
    Pixels beyond the boundaries of the original images have
    their values set to -1.
    """
    
    shifts = np.array([[0.0, 0.0] for k in range(len(raw))])
    aligned = [np.zeros_like(raw[0]) for k in range(len(raw))]
    aligned[0] = raw[0][::]
    
    for k in range(1, len(raw)):
        master.stackdisp.set('Calculating shifts... ' + str(k+1) + '/' + str(len(raw)))
        master.master.update_idletasks()
        shtmp = calculate_shift(raw[0], raw[k], pxwidth)
        shifts[k] += np.array(shtmp)
        aligned[k] = alignoneimage(raw[k], shifts[k])
    return [shifts, aligned]

def genmap(raw, shift):
    """Given two images and the shift between then, converts
    both to OD and then takes the difference of the two images
    to generate an elemental map."""
    thr_0 = threshold_otsu(raw[0])
    thr_1 = threshold_otsu(raw[1])
    
    fl_0 = np.ndarray.flatten(raw[0])
    I0_0 = np.mean([x for x in fl_0 if x > thr_0])
    fl_1 = np.ndarray.flatten(raw[1])
    I0_1 = np.mean([x for x in fl_1 if x > thr_1])
    
    od_0 = np.log(I0_0/raw[0])
    od_1 = alignoneimage(np.log(I0_1/raw[1]), shift)
    
    map_out = od_1 - od_0
    mask = 1.0 - 1.0*(od_1 == -1)
    
    return map_out*mask

def autoseg(dataset, bdy):
    """Given a dataset class (see sl_main for the definition),
    automatically segments using Otsu's method and returns
    regions with a boundary with width of the specified
    number of pixels."""

    thr = threshold_otsu(dataset.rawimg)
    
    i0_mask = (dataset.displayimg > thr)*(dataset.keeppx)
    if int(bdy/2) + bdy%2 > 0:
        i0_mask = 1*binary_erosion(i0_mask, iterations = int(bdy/2) + bdy%2, border_value = 1)
    
    it_mask = (dataset.displayimg < thr)*(dataset.keeppx)
    if int(bdy/2) > 0:
        it_mask = 1*binary_erosion(it_mask, iterations = int(bdy/2), border_value = 0)
    
    (a,b) = i0_mask.shape
    my_i0px = []
    my_itpx = []
    for i in range(a):
        for j in range(b):
            if i0_mask[i][j] == 1:
                my_i0px.append([i,j])
            if it_mask[i][j] == 1:
                my_itpx.append([i,j])
    
    return([my_i0px, my_itpx, i0_mask - it_mask])

def regridlinescan(rawimg, energies, dims):
    """Re-grids a line scan image for display with pyplot.imshow."""
    tmp = rawimg.T
    
    newgrid = [tmp[0]]
    
    en = energies[0]
    k = 1
    while k < len(energies):
        while en < energies[k]:
            newgrid = np.append(newgrid, [tmp[k-1]], axis = 0)
            en += 0.05
        k += 1
    
    newgrid = newgrid.T
    
    return(newgrid)

def predict_regions(rawimg, dims):
    """Given an image and dimensions, predicts an upsampled version
    and returns centroids of regions with (a) OD between 0.5 and 1.5
    and (b) area at least 1 um^2."""
    
    # Apply neural net upsampling
    imgin = np.array([[rawimg]]).astype(np.float32)
    usamp = upsampler.predict(imgin)[0,0]
        
    # Convert to OD
    thr = threshold_otsu(usamp)
    myi0 = np.mean(usamp[usamp>thr])
    usampOD = np.log(myi0/usamp)
    
    mask = (usampOD>0.5)*(usampOD<1.5)
    a = (4*rawimg.shape[0]-usampOD.shape[0])//2
    b = (4*rawimg.shape[1]-usampOD.shape[1])//2
    mask = np.pad(mask, ((a,a),(b,b)), mode='constant')
    reg = regionprops(label(mask))
    okcentroids = []
    
    for r in reg:
        scaledarea = r.area*dims[2]*dims[3]/16
        if scaledarea>1:
            okcentroids.append([r.centroid[0]*dims[2]/4,
                                r.centroid[1]*dims[3]/4,
                                scaledarea])
    
    okcentroids.sort(key=lambda ca: -ca[2])
    
    return okcentroids[:10]

if __name__ == '__main__':
    main()