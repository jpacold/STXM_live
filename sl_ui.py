import tkinter as tk
import tkinter.filedialog as fdialog

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigCanvas
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg as NavBar
from matplotlib.figure import Figure
from matplotlib import cm
from matplotlib import path as mplpath
from matplotlib import pyplot as plt

from os import path


class ImgFrame(tk.Frame):
    def on_click(self, event):
        if event.inaxes is not None and self.lassoactive:
            if event.button == 1:
                self.lassopts.append([event.xdata,
                                      event.ydata])
                self.redraw(self.master)
            if event.button == 3 or (len(self.lassopts) == 2 and self.master.mode == 'linescan'):
                self.lassoactive = False
                self.master.configure(cursor = 'arrow')
                self.master.addI0.configure(foreground = '#000000')
                self.master.addIT.configure(foreground = '#000000')
                self.lassofinish()
    
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.fig = Figure(figsize = (6.5, 5))
        self.img = self.fig.add_subplot(111)
        self.canvas = FigCanvas(self.fig, master = self)
        self.canvas._tkcanvas.config(highlightthickness=0)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side = 'top')
        
        # for ROI selection
        self.lassoactive = False
        self.lassopts = []
        self.lassoroiid = 0
        self.canvas.callbacks.connect("button_press_event", self.on_click)
        
        # for overlay showing I0/IT
        self.cdict = {'red'   : ((0.0, 0.0, 0.0),
                                 (0.5, 1.0, 1.0),
                                 (1.0, 0.867, 0.867)),
                      'green' : ((0.0, 1.0, 1.0),
                                 (0.5, 1.0, 1.0),
                                 (1.0, 0.719, 0.719)),
                      'blue'  : ((0.0, 1.0, 1.0),
                                 (0.5, 1.0, 1.0),
                                 (1.0, 0.527, 0.527)),
                      'alpha' : ((0.0, 1.0, 1.0),
                                 (0.5, 0.0, 0.0),
                                 (1.0, 1.0, 1.0))}
        plt.register_cmap(name = 'mycmap', data = self.cdict)
        self.mycmap = plt.get_cmap('mycmap')
    
    def redraw(self, master):
        self.img.clear()
        self.img.set_title(self.master.imgselect.get(), y = 1.01)
        
        self.img.tick_params(which = 'both', direction = 'out')
        self.img.get_xaxis().tick_bottom() 
        self.img.get_yaxis().tick_left()
        self.img.set_ylabel('$\mu$m')
        
        if master.mode == 'linescan':
            self.img.set_xlabel('Energy (eV)', labelpad = -3)
            self.img.imshow(self.master.data.rawstack[0], cmap = cm.gray, interpolation = 'nearest', aspect = 'auto',
                            extent = (self.master.data.energies[0], self.master.data.energies[-1], 0.0, self.master.data.imgdims[1]))
        
            if self.master.showrois.get() == 1:
                self.img.imshow(self.master.data.overlayimg, cmap=self.mycmap, alpha=0.5, interpolation='nearest', vmin = -1, vmax = 1,
                    extent = (self.master.data.energies[0], self.master.data.energies[-1], 0.0, self.master.data.imgdims[1]), aspect = 'auto')
        
        else:
            self.img.set_xlabel('$\mu$m', labelpad = -3)
            
            if master.autoalign.get() == 1:
                disp = self.master.data.displayimg
            else:
                disp = self.master.data.rawimg
            self.img.imshow(disp, cmap = cm.gray, interpolation = 'nearest',
                            extent = (0.0, self.master.data.imgdims[0], 0.0, self.master.data.imgdims[1]),
                            vmin = np.min(self.master.data.rawimg), vmax = np.max(self.master.data.rawimg))
            
            self.img.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(self.img.xaxis.get_majorticklocs()[1]))
            
            if self.master.showrois.get() == 1:
                self.img.imshow(self.master.data.overlayimg, cmap=self.mycmap, alpha=0.5,
                    extent = (0.0, self.master.data.imgdims[0], 0.0, self.master.data.imgdims[1]),
                    interpolation='nearest', vmin = -1, vmax = 1)
        
        if self.lassoactive:
            self.img.autoscale(False)
            if master.mode == 'linescan':
                for i in range(len(self.lassopts)):
                    self.img.plot([master.data.energies[0], master.data.energies[-1]], [self.lassopts[i][1], self.lassopts[i][1]], color = 'red')
            else:
                for i in range(len(self.lassopts) - 1):
                    self.img.plot([self.lassopts[i][0], self.lassopts[i+1][0]], [self.lassopts[i][1], self.lassopts[i+1][1]], color = 'red')
        
        self.canvas.draw()
    
    def lassofinish(self):
        (a, b) = self.master.data.overlayimg.shape
        
        if self.master.mode == 'linescan':
            for i in range(a):
                if self.master.data.overlayimg[a-i-1][0] == 0.0:
                    if i*self.master.data.imgdims[3] <= max(self.lassopts[0][1], self.lassopts[1][1]):
                        if i*self.master.data.imgdims[3] >= min(self.lassopts[0][1], self.lassopts[1][1]):
                            for j in range(b):
                                self.master.data.overlayimg[a-i-1][j] = self.lassoroiid
        
        else:
            lassopath = mplpath.Path(self.lassopts + [self.lassopts[-1]], closed = True)
            
            for i in range(a):
                for j in range(b):
                    if self.master.data.overlayimg[a-i-1][j] == 0.0 and lassopath.contains_point([j*self.master.data.imgdims[2],i*self.master.data.imgdims[3]]):
                        self.master.data.overlayimg[a-i-1][j] = self.lassoroiid
                        if self.lassoroiid == 1:
                            self.master.data.i0px.append([a-i-1,j])
                        else:
                            self.master.data.itpx.append([a-i-1,j])
                            self.master.data.backupit = self.master.data.itpx[:]
                
        self.redraw(self.master)
        self.master.enablectrls()
        
        self.master.genI0IT()


class SpecFrame(tk.Frame):
    def on_click(self, event):
        if event.inaxes is not None and event.button == 3:
            if self.master.mode == 'stack':
                en = min(event.xdata, self.master.data.energies[-1])
                i = 0
                while self.master.data.energies[i] < en:
                    i += 1
                self.master.setxim(min(i, len(self.master.data.rawstack) - 1))
    
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.fig = Figure(figsize = (6.5, 5))
        self.spec = self.fig.add_subplot(111)
        self.spec.cbar = None
        self.canvas = FigCanvas(self.fig, master = self)
        self.canvas._tkcanvas.config(highlightthickness=0)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side = 'top')
        
        self.canvas.callbacks.connect("button_press_event", self.on_click)
    
    def replotspec(self):
        self.spec.clear()
        if self.spec.cbar:
            self.fig.clear()
            self.spec = self.fig.add_subplot(111)
        self.spec.set_title(path.basename(self.master.hdrfile)[:-4], y = 1.01)
        self.spec.set_xlabel('Energy (eV)')
        
        if self.master.plotrb.get() == 0:
            plotdata = self.master.data.i0
            self.spec.set_ylabel('Average I0', labelpad = 0)
        elif self.master.plotrb.get() == 1:
            plotdata = self.master.data.it
            self.spec.set_ylabel('Average I', labelpad = 0)
        elif self.master.plotrb.get() == 2:
            plotdata = self.master.data.od
            self.spec.set_ylabel('OD', labelpad = 0)

        self.spec.plot(self.master.data.energies[:len(plotdata)], plotdata)
        
        self.canvas.draw()

    def showmap(self):
        self.spec.clear()
        if self.spec.cbar:
            self.fig.clear()
            self.spec = self.fig.add_subplot(111)
        self.spec.set_title(path.basename(self.master.hdrfile)[:-4] + ' Map', y = 1.01)
        
        self.spec.tick_params(which = 'both', direction = 'out')
        self.spec.get_xaxis().tick_bottom()
        self.spec.get_yaxis().tick_left()
        self.spec.set_xlabel('$\mu$m', labelpad = -3)
        self.spec.set_ylabel('$\mu$m')
        
        map_show = self.spec.imshow(self.master.data.eltmap, cmap = cm.gray, interpolation = 'nearest',
                                    extent = (0.0, self.master.data.imgdims[0], 0.0, self.master.data.imgdims[1]))
        self.spec.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(self.spec.xaxis.get_majorticklocs()[1]))
        self.spec.autoscale(False)
        self.spec.cbar = self.fig.colorbar(map_show)
        self.spec.cbar.set_label('$\\Delta{}$OD')
        
        self.canvas.draw()

def makefilepicker(master):
    master.filepickframe = tk.Frame(master)
    
    master.hdrbox = tk.Entry(master.filepickframe, width = 80)
    master.hdrbox.insert(0, master.hdrfile)
    master.hdrbox.pack(side = 'left', padx = master.gridpad, pady = master.gridpad)
    
    master.browsebtn = tk.Button(master.filepickframe, text = "Select .hdr file", padx = master.btnpad, pady = master.btnpad)
    master.browsebtn.pack(side = 'left', padx = master.gridpad, pady = master.gridpad)
    
    master.recentbtn = tk.Button(master.filepickframe, text = "Most recent", padx = master.btnpad, pady = master.btnpad)
    master.recentbtn.pack(side = 'left', padx = master.gridpad, pady = master.gridpad)
    
    master.filepickframe.grid(row = 0, column = 0, sticky = 'NSW', columnspan = 7, padx = master.btnpad, pady = master.btnpad)
    
def makeimgcontrols(master):
    #master.spacer1 = tk.Label(master, text = "", padx = master.btnpad, pady = 2)
    #master.spacer1.grid(row = 1, column = 0, columnspan = 8)
    
    # ---------------------------------------------------------------------
    # ROI selection tools
    master.addI0 = tk.Button(master, text = "Add region to I0", padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.addI0.grid(row = 2, column = 0, sticky = 'W', padx = master.gridpad, pady = master.gridpad)
    master.addIT = tk.Button(master, text = "Add region to I", padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.addIT.grid(row = 3, column = 0, sticky = 'W', padx = master.gridpad, pady = master.gridpad)
    
    master.clearI0 = tk.Button(master, text = "Clear I0", padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.clearI0.grid(row = 2, column = 1, sticky = 'W', padx = master.gridpad, pady = master.gridpad)
    master.clearIT = tk.Button(master, text = "Clear I", padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.clearIT.grid(row = 3, column = 1, sticky = 'W', padx = master.gridpad, pady = master.gridpad)
    
    master.spacer2 = tk.Label(master, text = "", padx = master.btnpad, pady = 0)
    master.spacer2.grid(row = 4, column = 0, columnspan = 2)
    
    master.autosegment = tk.Button(master, text = "Auto-select I0 and I", padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.autosegment.grid(row = 5, column = 0, sticky = 'W', padx = master.gridpad, pady = master.gridpad)
    
    # ---------------------------------------------------------------------
    # Omit pixels near boundary
    master.bdyframe = tk.Frame(master)
    
    master.bdytext = tk.Label(master.bdyframe, text = "Boundary width (px):")
    master.bdytext.pack(side = 'left')
    
    master.bdyentry = tk.Entry(master.bdyframe, width = 2)
    master.bdyentry.insert(0, '2')
    master.bdyentry.pack(side = 'left')
    
    master.bdyframe.grid(row = 6, column = 0, sticky = 'W', padx = master.gridpad, pady = master.gridpad - 1)
    
    # ---------------------------------------------------------------------
    # legend (plain graphics on Tk canvas)
    master.legend = tk.Canvas(master, width = 125, height = 100, background = 'white', bd = 0)
    
    master.legend.create_rectangle(3,3,124,99, fill = 'white', outline = 'black')
    master.legend.create_text(62,12, text = 'Image Legend')
    
    cspec = ['#000000', '#555555', '#AAAAAA']
    for x in range(len(cspec)):
        master.legend.create_rectangle(10*x+8,30,10*x+18,40, fill  = cspec[x], outline = '' )
    master.legend.create_text(50,35, text = 'Raw image', anchor = 'w')
    
    cspec = ['#4A3D2D', '#947A5A', '#DEB887']
    for x in range(len(cspec)):
        master.legend.create_rectangle(10*x+8,55,10*x+18,65, fill  = cspec[x], outline = '' )
    master.legend.create_text(50,60, text = 'I0', anchor = 'w')
    
    cspec = ['#005555', '#00AAAA', '#00FFFF']
    for x in range(len(cspec)):
        master.legend.create_rectangle(10*x+8,80,10*x+18,90, fill  = cspec[x], outline = '' )
    master.legend.create_text(50,85, text = 'I', anchor = 'w')
    
    # cspec = ['#004400', '#006600', '#008800']
    # for x in range(len(cspec)):
    #     master.legend.create_rectangle(10*x+8,105,10*x+18,115, fill  = cspec[x], outline = '' )
    # master.legend.create_text(50,110, text = 'I (omitted from spectrum)', anchor = 'w')
    
    master.legend.grid(row = 2, column = 2, rowspan = 5, columnspan = 1, padx = 0, pady = 0, sticky = 'NW')
    
    # ---------------------------------------------------------------------
    # Controls for limiting the OD of pixels in the I region
    master.odlimit = tk.Button(master, text = "Filter I selection", padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.odlimit.grid(row = 7, sticky = 'W', padx = master.gridpad, pady = master.gridpad)
    
    master.odframe = tk.Frame(master)
    
    master.spacerod = tk.Label(master.odframe, text = "", padx = 4, pady = 2)
    master.spacerod.pack(side = 'left')
    
    master.odmin = tk.Entry(master.odframe, width = 4)
    master.odmin.insert(0, '0.80')
    master.odmin.pack(side = 'left')
    
    master.odlbl = tk.Label(master.odframe, text = "< OD <")
    master.odlbl.pack(side = 'left')
    
    master.odmax = tk.Entry(master.odframe, width = 4)
    master.odmax.insert(0, '2.00')
    master.odmax.pack(side = 'left')
    
    master.odframe.grid(row = 8, column = 0, sticky = 'W', padx = master.gridpad, pady = master.gridpad)
    
    # ---------------------------------------------------------------------
    # Display and stack options
    master.showrois = tk.IntVar()
    master.showroischk = tk.Checkbutton(master, variable = master.showrois, text = 'Show ROI overlay', padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.showroischk.grid(row = 6, column = 2, sticky = 'W', padx = master.gridpad, pady = master.gridpad)
    
    master.autoalign = tk.IntVar()
    master.autoalignchk = tk.Checkbutton(master, variable = master.autoalign, text = 'Align images in stack', padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.autoalignchk.grid(row = 7, column = 2, sticky = 'W', padx = master.gridpad, pady = master.gridpad)
    
    master.suggestnext = tk.Button(master, text = 'Suggest regions to inspect', padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.suggestnext.grid(row = 8, column = 2, sticky = 'W', padx = master.gridpad, pady = master.gridpad)
    
    master.imgnavframe = tk.Frame(master)
    master.imgselect = tk.StringVar()
    master.imgselect.set('          ')
    #master.imgmenu = tk.OptionMenu(master.imgnavframe, master.imgselect, '              ')
    #master.imgmenu.config(width = 35)
    #master.imgmenu.grid(row = 0, column = 2, padx = master.gridpad, pady = master.gridpad)
    
    master.imgfirst = tk.Button(master.imgnavframe, text = "|<", padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.imgfirst.grid(row = 0, column = 0, padx = master.gridpad, pady = master.gridpad, sticky = 'E')
    master.imgback = tk.Button(master.imgnavframe, text = "<", padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.imgback.grid(row = 0, column = 1, padx = master.gridpad, pady = master.gridpad, sticky = 'W')
    master.imgnext = tk.Button(master.imgnavframe, text = ">", padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.imgnext.grid(row = 0, column = 3, padx = master.gridpad, pady = master.gridpad, sticky = 'E')
    master.imglast = tk.Button(master.imgnavframe, text = ">|", padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.imglast.grid(row = 0, column = 4, padx = master.gridpad, pady = master.gridpad, sticky = 'W')
    
    master.imgnavframe.grid(row = 11, column = 0, columnspan = 3, padx = 0, pady = 0)

def makespectrumcontrols(master):
    # ---------------------------------------------------------------------
    # Choose what channel to plot
    master.plotlbl = tk.Label(master, text = "Plot:")
    master.plotlbl.grid(row = 2, column = 3, sticky = 'W', padx = 10 + master.gridpad, pady = master.gridpad)
    
    master.plotrb = tk.IntVar()
    master.ploti0 = tk.Radiobutton(master, text = "I0", padx = master.btnpad, variable = master.plotrb, value = 0, state = tk.DISABLED)
    master.ploti0.grid(row = 3, column = 3, sticky = 'W', padx = master.gridpad + 20, pady = master.gridpad)
    master.plotit = tk.Radiobutton(master, text = "I", padx = master.btnpad, variable = master.plotrb, value = 1, state = tk.DISABLED)
    master.plotit.grid(row = 4, column = 3, sticky = 'W', padx = master.gridpad + 20, pady = master.gridpad)
    master.plotod = tk.Radiobutton(master, text = "OD", padx = master.btnpad, variable = master.plotrb, value = 2, state = tk.DISABLED)
    master.plotod.grid(row = 5, column = 3, sticky = 'W', padx = master.gridpad + 20, pady = master.gridpad)
    
    # ---------------------------------------------------------------------
    # Save buttons
    master.savestack = tk.Button(master, text = "Save stack", padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.savestack.grid(row = 2, column = 4, sticky = 'W', padx = master.gridpad, pady = master.gridpad)
    master.stackfname = tk.Entry(master, width = 20)
    master.stackfname.grid(row = 2, column = 5, sticky = 'W', padx = master.gridpad, pady = master.gridpad)
    master.stackfname.insert(0, '.ncb')
    
    master.savespectrum = tk.Button(master, text = "Save spectrum", padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.savespectrum.grid(row = 3, column = 4, sticky = 'W', padx = master.gridpad, pady = master.gridpad)
    master.spectrumfname = tk.Entry(master, width = 20)
    master.spectrumfname.grid(row = 3, column = 5, sticky = 'W', padx = master.gridpad, pady = master.gridpad)
    master.spectrumfname.insert(0, '.txt')
    
    master.saveI0 = tk.Button(master, text = "Save I0", padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.saveI0.grid(row = 4, column = 4, sticky = 'W', padx = master.gridpad, pady = master.gridpad)
    master.I0fname = tk.Entry(master, width = 20)
    master.I0fname.grid(row = 4, column = 5, sticky = 'W', padx = master.gridpad, pady = master.gridpad)
    master.I0fname.insert(0, '_i0.txt')
    
    master.savealn = tk.Button(master, text = "Save alignment", padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.savealn.grid(row = 5, column = 4, sticky = 'W', padx = master.gridpad, pady = master.gridpad)
    master.alnfname = tk.Entry(master, width = 20)
    master.alnfname.grid(row = 5, column = 5, sticky = 'W', padx = master.gridpad, pady = master.gridpad)
    master.alnfname.insert(0, '.aln')
    
    master.saveallframe = tk.Frame(master)
    
    master.saveall = tk.Button(master.saveallframe, text = "Save all", padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.saveall.pack(side = 'left', padx = master.gridpad, pady = master.gridpad)
    
    master.autosave = tk.IntVar()
    master.autosave.set(1)
    master.autosavechk = tk.Checkbutton(master.saveallframe, variable = master.autosave, text = "Autosave all when stack is complete", padx = master.btnpad, pady = master.btnpad, state = tk.DISABLED)
    master.autosavechk.pack(side = 'left', padx = 20 + master.gridpad)
    
    master.saveallframe.grid(row = 6, column = 4, sticky = 'W', pady = master.gridpad, columnspan = 3)
    
    master.filedisp = tk.StringVar()
    master.filestatus = tk.Label(master, text = "", textvariable = master.filedisp, background = '#00008B',
                                 foreground = 'white', anchor = 'w', padx = master.btnpad, pady = master.btnpad)
    master.filestatus.grid(row = 7, column = 3, sticky = 'EW', padx = 5 + master.gridpad, columnspan = 5)
    
    master.stackdisp = tk.StringVar()
    master.stackstatus = tk.Label(master, text = "", textvariable = master.stackdisp, background = '#006400',
                                  foreground = 'white', anchor = 'w', padx = master.btnpad, pady = master.btnpad)
    master.stackstatus.grid(row = 8, column = 3, sticky = 'EW', padx = 5 + master.gridpad, columnspan = 5)

def makemplpanels(master):
    # Image display
    master.imgdisplay = ImgFrame(master)
    master.imgdisplay.grid(row = 10, column = 0, sticky = 'NESW', columnspan = 3, padx = master.gridpad, pady = master.gridpad)
    
    # Spectrum display
    master.specdisplay = SpecFrame(master)
    master.specdisplay.grid(row = 10, column = 3, sticky = 'NESW', columnspan = 5, padx = master.gridpad, pady = master.gridpad)
    
    master.specnavframe = tk.Frame(master = master)
    master.specnavframe.navbar = NavBar(master.specdisplay.canvas, master.specnavframe)
    master.specnavframe.navbar.pack(anchor = 'w')
    master.specnavframe.navbar.update()
    master.specnavframe.grid(row = 11, column = 3, sticky = 'NESW', columnspan = 5, padx = master.gridpad, pady = master.gridpad)

if __name__ == '__main__':
    main()