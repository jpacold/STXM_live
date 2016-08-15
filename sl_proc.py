import numpy as np

#from scipy.ndimage import fourier_shift
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import shift #, zoom
from scipy.ndimage.morphology import binary_erosion

from skimage.feature import register_translation
from skimage.filters import sobel

def otsu(img):
    """Calculates a threshold for a integer-
    valued image using Otsu's method."""
    
    tmp = np.ndarray.flatten(img.astype(np.int))
    
    mymin = np.min(tmp)
    mymax = np.max(tmp)
    myavg = np.mean(tmp)
    
    bins = np.array([n for n in range(mymin, mymax+1)])
    vals = np.zeros_like(bins, dtype = np.float)
    for x in tmp:
        vals[x-mymin] += 1.0
    vals /= np.sum(vals)
    
    variance = np.zeros_like(bins, dtype = np.float)
    wlow = vals[0]
    mulow = mymin
    whigh = 1.0 - wlow
    muhigh = (myavg - wlow*mulow)/whigh
    variance[0] = wlow*whigh*(mulow - muhigh)*(mulow - muhigh)
    
    for n in range(1, mymax - mymin):
        wlow += vals[n]
        mulow = np.sum( bins[:n]*vals[:n] )/np.sum(vals[:n])
        whigh = 1.0 - wlow
        muhigh = (myavg - wlow*mulow)/whigh
        variance[n] = wlow*whigh*(mulow - muhigh)*(mulow - muhigh)
    
    return np.float(mymin + np.argmax(variance))

def calculate_shift(imgA, imgB, pxwidth):
    """Uses the registration algorithm from scikit-image
    to calculate alignment to within 0.01 pixels or 1 nm,
    whichever is coarser."""
    
    sigma = 0.15/pxwidth
    filtA = gaussian_filter(imgA, min(sigma, 3.0))
    filtB = gaussian_filter(imgB, min(sigma, 3.0))
    
    thr = otsu(filtB)
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

# def calculate_shift_scipy(imgA, imgB, pxwidth, zfmax = None):
#     """Given two images and the width of each pixel (in microns),
#     applies a Gaussian filter to both images, using a Gaussian
#     with a width of 0.15 microns (i.e., approximately 3 times
#     the spatial resolution of a typical STXM image). Calculates
#     the 2D cross-correlation and returns the shift that maximizes
#     the cross-correlation, upsampling to calculate the shift to
#     within 10 nanometers or 1/10 pixel, whichever is finer.
#     """
#     
#     # Apply Gaussian filter
#     sigma = 0.15/pxwidth
#     filtA = gaussian_filter(imgA, sigma, mode = 'nearest')
#     filtB = gaussian_filter(imgB, sigma, mode = 'nearest')
#     
#     # Calculate shift
#     if zfmax is None:
#         lim = max(1.0, 10.0, pxwidth/0.01)
#     else:
#         lim = zfmax
# 
#     if lim == 1.0:
#         tmpA = filtA[::]
#         tmpB = filtB[::]
#     else:
#         zf = lim/2.0
#         tmpA = zoom(filtA, zf)
#         tmpB = zoom(filtB, zf)
#     
#     image_product = np.fft.fft2(tmpA) * np.fft.fft2(tmpB).conj()
#     corr = np.fft.fftshift(np.fft.ifft2(image_product))            
#         
#     s = corr.shape
#     origin = [(q - (q%2))/2 for q in s]
#     
#     x = np.argmax(corr.real)
#     center = [int(x/s[1]), x%s[1]]
# 
#     finalsh = [np.float(origin[0] - center[0])/zf, np.float(origin[1] - center[1])/zf]
#     print(lim, zf, x, finalsh)
#     
#     if lim != 1.0:
#         tmpA = zoom(tmpA, 2.0)
#         tmpB = zoom(shift(tmpB, [-finalsh[0], -finalsh[1]], mode = 'nearest'), 2.0)
#         me = np.mean(tmpA)
#         st = np.std(tmpA)
#         nrmA = (tmpA - me)/st
#         
#         corr = np.zeros((3,3), dtype = np.float)
#         for ii in range(3):
#             rollB = np.roll(tmpB, ii - 1, axis = 0)
#             for jj in range(3):
#                 nrmB = np.roll(rollB, jj - 1, axis = 1)
#                 corr[ii][jj] += np.sum(nrmA*nrmB)
#         
#         x = np.argmax(corr)
#         center = [int(x/3), x%3]
#         
#         finalsh[0] += np.float(1 - center[0])/lim
#         finalsh[1] += np.float(1 - center[1])/lim
#     
#     
#     return finalsh

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
    thr_0 = otsu(raw[0])
    thr_1 = otsu(raw[1])
    
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

    thr = otsu(dataset.rawimg)
    
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

if __name__ == '__main__':
    main()