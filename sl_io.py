import numpy as np

from glob import glob
from os import path


def loadstack(hdrfile, master):
    """Takes the full path to a STXM .hdr file as input.
    Parses the .hdr file to get the energy points and the
    dimensions of the image(s) and pixels.
    
    Builds a list of all existing raw image files associated
    with the .hdr file and then loads each file.
    
    Returns the list of energies, the image dimensions, a stack
    of raw images, and a list of existing image files.
    """
    
    with open(hdrfile, 'r') as hdr:
        ln = hdr.readline()
        
        while 'ScanDefinition' not in ln:
            ln = hdr.readline()
        scantype = ln.split(';')[1][9:-1]
        
        while 'Axis = { Name = \"Energy\"' not in ln:
            ln = hdr.readline()
                
        ln = hdr.readline()
        energies = ln[:ln.index(')')]
        energies = energies.split(', ')
        energies = [float(en) for en in energies[1:]]
            
        
        if "Line Scan" in scantype:
            while 'Axis = { Name = \"Sample\"' not in ln:
                ln = hdr.readline()
            
            ln = ln.split()
            dims = [energies[-1] - energies[0], float(ln[ln.index('Max')+2][:-1]) - float(ln[ln.index('Min')+2][:-1]), 1.0, 1.0 ]
            ln = hdr.readline()
            ln = ln.split()
            dims[2] = int(ln[ln.index('Points')+2][1:-1])
            dims[3] = dims[1]/float(dims[2])
        
        else:
            while 'XRange = ' not in ln:
                ln = hdr.readline()
            
            ln = ln.split()
            dims = [ln[ln.index(s) + 2] for s in ['XRange', 'YRange', 'XStep', 'YStep']]
            dims = [float(d[:-1]) for d in dims]
    
    workdir = path.dirname(hdrfile)
    prefix = path.basename(hdrfile)[:-4]
    ximlist = glob(path.join(workdir, prefix) + '*.xim')
    ximlist.sort()
    ximlist = [path.abspath(f) for f in ximlist]
    raw = []
    for k in range(len(ximlist)):
        master.stackdisp.set('Loading image ' + str(k+1) + '/' + str(len(ximlist)))
        master.master.update_idletasks()
        raw.append(np.loadtxt(ximlist[k])[::-1])
    
    return [energies, dims, raw, ximlist, scantype]

def writencb(master):
    """Writes stack files (.ncb and .dat) in aXis2000 format."""
    
    fname = path.join(path.dirname(master.hdrfile), master.stackfname.get())
    hdrfile = master.hdrfile
    rawstack = master.data.rawstack
    energies = master.data.energies
    
    with open(hdrfile, 'r') as hdr:
        ln = hdr.readline()
        
        while ln[:12] != '{ CentreXPos':
            ln = hdr.readline()
        params = ln.split(';')
        
        xdim = float(params[2][9:])
        ydim = float(params[3][9:])
        xnpx = int(params[6][10:])
        ynpx = int(params[7][10:])
        
        ringcurrent = []
        imgids = []
        
        hdr.readline()
        while ln:
            ln = hdr.readline()
            if ln[:5] == 'Image':
                params = ln.split('; ')
                imgids.append(params[0][5:8])
                ringcurrent.append(float(params[0][-6:]))
        
    imgnames = [hdrfile[:-4] + '_a' + n + '.xim' for n in imgids]
    print(len(imgnames), len(ringcurrent))
    # normalize images to ring current 500.0
    for i in range(len(rawstack)):
        rawstack[i] *= 500.0
        rawstack[i] /= ringcurrent[i]
    
    alldata = np.array([])
    for im in rawstack:
        alldata = np.append(alldata, np.ndarray.flatten(im[::-1]))
    
    # check max counts. if possible, scale up by power of 10,
    # but keep the max counts below 32768 so that output can
    # be stored as 16-bit integers. if max counts are above
    # 32768, scale down by power of 10.
    
    m = np.max(alldata)
    scaleexp = np.log10(32767.0/m)
    if scaleexp > 0:
        scaleexp = int(scaleexp)
    else:
        scaleexp = int(scaleexp - 1)
    scale = (10.0)**scaleexp
    
    alldata *= scale
    
    # write .ncb -- just a list of 16-bit integers
    (alldata.astype('int16')).tofile(fname)
    
    # write .dat file associated with .ncb. The pattern is:
    #
    #         [# of pixels in x]   [# of pixels in y]    [count scale factor]
    #         0.00000              [x_width in microns]
    #         0.00000              [y_width in microns]
    #         [# of images]
    #         [energy, image 1   ]
    #         [energy, image 2   ]
    #         [    ...           ]
    #         [energy, last image]
    #[filename, image 1]     [energy, image 1   ]   2.00  (<--- not sure what the function of this is!)
    #[filename, image 2]     [energy, image 2   ]   2.00
    #[    ...                                   ]   2.00
    #[filename, last image]  [energy, last image]   2.00
    #
    
    datfile = open(fname[:-3] + 'dat', 'w')

    datfile.write(' '*(12 - len(str(xnpx))) + str(xnpx))
    datfile.write(' '*(12 - len(str(ynpx))) + str(ynpx))
    datfile.write(' '*(13 - len(str(scale))) + str(scale))
    datfile.write('\n')
    
    datfile.write('     0.000000')
    datfile.write(' '*(13 - len(str(xdim))) + str(xdim) + '\n')
    datfile.write('     0.000000')
    datfile.write(' '*(13 - len(str(ydim))) + str(ydim) + '\n')
    
    datfile.write(' '*(12 - len(str(len(rawstack)))) + str(len(rawstack)) + '\n')
    
    for i in range(len(rawstack)):
        datfile.write(' '*(13 - len(str(energies[i]))) + str(energies[i]) + '\n')

    for i in range(len(rawstack)):
        datfile.write(imgnames[i][-21:] + '  ' + str(energies[i]))
        if str(energies[i])[-2] == '.':
            datfile.write('0')
        datfile.write('   2.00\n')
    
    datfile.close()
    
    master.filedisp.set('Wrote stack file to ' + fname)

def writetxt(master, is_i0 = False):
    
    if is_i0:
        fname = path.join(path.dirname(master.hdrfile), master.I0fname.get())
        yvals = master.data.i0
    else:
        fname = path.join(path.dirname(master.hdrfile), master.spectrumfname.get())
        yvals = master.data.od
    
    energies = master.data.energies
    
    outfile = open(fname, 'w')
    outfile.write('% 1d\n')
    outfile.write('% ' + fname[-16:-4] + '\n')
    outfile.write('%  ' + str(len(yvals)) + '\n')
    outfile.write('% from stack ' + (path.dirname(fname))[-12:] + '\n')
    for i in range(len(yvals)):
        outfile.write('\t' + str(energies[i]) + '\t' + str(yvals[i]) + '\n')
    outfile.close()
    
    if is_i0:
        master.filedisp.set('Wrote I0 to ' + fname)
    else:
        master.filedisp.set('Wrote spectrum to ' + fname)

def writealn(master):
    fname = path.join(path.dirname(master.hdrfile), master.alnfname.get())
    
    fout = open(fname, 'w')
    
    fout.write('! Alignment file generated by STXM Live Analysis\n')
    fout.write('! X-Y Pixel shifts after alignment\n')
    fout.write('! Full images used\n')
    fout.write('! Aligned to first image, ' + path.basename(master.data.imgfile[0]) + '  ' + str(master.data.energies[0]) + ' eV\n')
    fout.write('! Correlation maximum determined by peak\n')
    fout.write('! No edge enhancement\n')
    fout.write('! Upsample factor 1000\n')
    fout.write('! Gaussian smoothing of 3 pixels\n')
    fout.write('ALIGN(0,0,0,0,1000,0.001,3,0,0,0,0,-1\n')
    fout.write('PLOTIT(' + str(len(master.data.shifts)) + '\n')
    
    for i in range(len(master.data.shifts)):
        fout.write(path.basename(master.data.imgfile[i]) + '  ' + str(master.data.energies[i]) + '   ' + '2.00,')
        fout.write(str(master.data.shifts[i][0]) + ',' + str(master.data.shifts[i][1]) + '\n')
    
    master.filedisp.set('Wrote alignment file to ' + fname)
    master.update_idletasks()

    fout.close()