import tkinter as tk
import tkinter.filedialog as fdialog

import numpy as np

from time import time
from threading import Timer

from os import path
from platform import system
from sl_ui import *
from sl_io import *
from sl_proc import *


class stxmdata():        
    def clear(self):
        self.rawstack = []
        self.alnstack = []
        self.rawimg = np.array([[0,0],[0,0]])
        self.displayimg = np.array([[0,0],[0,0]])
        self.overlayimg = np.array([[0,0],[0,0]])
        self.imgdims = [1,1,1,1]  # [XRange, YRange, XStep, YStep]
        self.imgfile = []
        self.shifts = []
        self.energies = [0,0]
        self.i0 = [0,0]
        self.it = [0,0]
        self.od = [0,0]
        self.i0px = []
        self.itpx = []
        self.backupit = []
        self.bdypx = []
        self.keeppx = np.array([[0,0],[0,0]])
    
    def __init__(self):
        self.clear()


class MainWindow(tk.Frame):

    def hdrdialog(self):
        tmphdr = fdialog.askopenfilename(filetypes = [(".hdr files", '*.hdr')], initialdir = path.dirname(self.hdrfile))
        if tmphdr:
            self.sethdr(tmphdr)
    
    def setxim(self, ind = None):
        if ind is None:
            if self.mode == 'single' or self.mode == 'linescan':
                i = 0
            else:
                currimg = self.imgselect.get()
                currimg = path.join(path.dirname(self.hdrfile), currimg.split()[0])
                i = self.data.imgfile.index(currimg)
        elif ind == 'next':
            i = min(len(self.data.rawstack) - 1, self.imglabels.index(self.imgselect.get()) + 1)
        elif ind == 'back':
            i = max(0, self.imglabels.index(self.imgselect.get()) - 1)
        else:
            i = ind
        
        self.data.rawimg = self.data.rawstack[i][::]
        self.imgselect.set(self.imglabels[i])
        
        if self.mode == 'single' or self.autoalign.get() == 0 or self.mode == 'linescan':
            self.data.displayimg = self.data.rawimg[::]
        else:
            self.data.displayimg = self.data.alnstack[i][::]
        
        self.imgdisplay.redraw(self)

    def sethdr(self, hfile):
        self.data.clear()
        self.filedisp.set('')
        
        self.hdrfile = path.abspath(hfile)
        self.workdir = path.dirname(self.hdrfile)
        self.hdrbox.delete(0, 'end')
        self.hdrbox.insert(0, self.hdrfile)
        
        self.starttime = path.getctime(self.hdrfile)
        
        self.stackdisp.set('Loading...')
        self.data.energies, self.data.imgdims, self.data.rawstack, self.data.imgfile, self.data.scantype = loadstack(hfile, self)
        self.data.estrlen = max([len(str(x)) for x in self.data.energies])
        self.data.rawimg = self.data.rawstack[-1]
        self.data.overlayimg = np.zeros_like(self.data.rawimg)
        
        if "Line Scan" in self.data.scantype:
            self.mode = 'linescan'
        elif len(self.data.energies) == 1:
            self.mode = 'single'
        elif len(self.data.energies) == 2:
            self.mode = 'map'
        else:
            self.mode = 'stack'
        
        if self.mode == 'linescan':
            self.imglabels = [path.basename(self.data.imgfile[0]) + '     Line Scan']
        else:            
            self.imglabels = []
            for i in range(len(self.data.imgfile)):
                zeropad = self.data.estrlen - len(str(self.data.energies[i]))
                f = path.basename(self.data.imgfile[i]) + '     ' + str(self.data.energies[i]) + '0'*zeropad + ' eV'
                self.imglabels.append(f)
        
        if self.mode == 'linescan':
            self.data.alnstack = [regridlinescan(self.data.rawstack[0], self.data.energies, self.data.imgdims)]
            self.data.overlayimg = np.zeros_like(self.data.alnstack[0])
        
        if self.mode == 'map' and len(self.data.rawstack) == 2:
            self.stackdisp.set('Generating map...')
            self.data.shifts = [[0,0], calculate_shift(self.data.rawstack[0], self.data.rawstack[1], self.data.imgdims[3])]
            self.data.alnstack = [self.data.rawstack[0], alignoneimage(self.data.rawstack[1], self.data.shifts[1])]
            self.data.eltmap = genmap(self.data.rawstack, self.data.shifts[1])
            self.specdisplay.showmap()
            self.stackdisp.set('Map complete')
        
        if self.mode == 'stack':
            # Align stack
            [self.data.shifts, self.data.alnstack] = alignstack(self.data.rawstack, self.data.imgdims[3], self)
            # Keep track of which pixels have
            # drifted out of field of view
            self.data.keeppx = np.ones_like(self.data.rawstack[0])
            for im in self.data.alnstack:
                mask = 1 - 1*(im == -1)
                self.data.keeppx *= mask
            
            self.data.i0 = np.zeros(len(self.data.rawstack))
            self.data.it = np.zeros(len(self.data.rawstack))
            self.specdisplay.replotspec()
        
        self.setxim(ind = len(self.data.imgfile) - 1)
        self.stackdisp.set('')

        self.data.keeppx = np.ones_like(self.data.rawstack[0])
        if self.mode == 'map' or self.mode == 'stack':
            # Keep track of pixels that are okay to keep after alignment
            for k in range(len(self.data.rawstack)):
                self.data.keeppx *= (self.data.alnstack[-1] != -1)
            
            if len(self.data.energies) > len(self.data.rawstack):
                self.stackrunning = True
                self.chkpoint = time()
                self.livestack()
            else:
                self.stackrunning = False
                if self.refresh is not None:
                    self.refresh.cancel()
        else:
            self.stackrunning = False
            if self.refresh is not None:
                self.refresh.cancel()
        
        self.setfnames()
        self.enablectrls()
    
    def mostrecent(self):
        if self.hdrfile[-4:] == '.hdr':
            if self.mode == 'single':
                tmpdir = path.dirname(self.hdrfile)
            else:
                tmpdir = path.dirname(path.dirname(self.hdrfile)[:-1])
            
            hdrlist = glob(path.join(tmpdir, '*.hdr')) + glob(path.join(tmpdir, '*', '*.hdr'))
            hdrlist = [ (path.basename(h), h) for h in hdrlist ]
            hdrlist.sort()
            
            newhdrfile = hdrlist[-1][1]
            
            self.sethdr(path.abspath(newhdrfile))
           
    def addtoroi(self, roiid):
        self.config(cursor = 'plus')
        self.imgdisplay.lassopts = []
        self.imgdisplay.lassoactive = True
        self.imgdisplay.lassoroiid = roiid

        for btn in [self.addI0, self.clearI0, self.addIT, self.clearIT, self.autosegment]:
                btn.configure(state = tk.DISABLED)
        if roiid == 1:
            self.addI0.configure(foreground = '#DEB887')
        else:
            self.addIT.configure(foreground = '#00FFFF')
    
    def autoselect_rois(self):
        if self.mode == 'linescan':
            pass
        
        else:
            self.data.i0px, self.data.itpx, self.data.overlayimg = autoseg(self.data, int(self.bdyentry.get()))
            self.data.backupit = self.data.itpx[:]
        
        self.genI0IT()
        
    def clearroi(self, roiid):
        if roiid == 1:
            self.data.i0px = []
        else:
            self.data.itpx = []
        self.data.backupit = []
        
        mask = roiid*(self.data.overlayimg == roiid)
        self.data.overlayimg -= mask
                
        self.imgdisplay.redraw(self)
        self.genI0IT()
    
    def odfilter(self):
        if len([x for x in self.data.od if x != 0.0]) > 0:
            curr_i0 = np.mean(np.array([self.data.displayimg[p[0]][p[1]] for p in self.data.i0px]))
            odminv = float(self.odmin.get())
            odmaxv = float(self.odmax.get())
            
            self.data.itpx = [p for p in self.data.backupit if np.log(curr_i0/self.data.displayimg[p[0]][p[1]]) < odmaxv]
            self.data.itpx = [p for p in self.data.itpx if np.log(curr_i0/self.data.displayimg[p[0]][p[1]]) > odminv]
            
            for p in self.data.backupit:
                if p in self.data.itpx:
                    self.data.overlayimg[p[0]][p[1]] = -1
                else:
                    self.data.overlayimg[p[0]][p[1]] = 0
        
        self.genI0IT()
    
    def genI0IT(self):
        
        if self.mode == 'stack' or self.mode == 'single':
            if self.autoalign.get() == 0 or self.mode == 'single':
                my_stack = self.data.rawstack
                my_i0px = self.data.i0px
                my_itpx = self.data.itpx
            else:
                my_stack = self.data.alnstack
                my_i0px = [p for p in self.data.i0px if self.data.keeppx[p[0]][p[1]] == 1]
                my_itpx = [p for p in self.data.itpx if self.data.keeppx[p[0]][p[1]] == 1]
            
            self.data.i0 = np.zeros(len(my_stack))
            self.data.it = np.zeros(len(my_stack))
            
            for i in range(len(my_stack)):
                for p in my_i0px:
                    self.data.i0[i] += my_stack[i][p[0]][p[1]]
                for p in my_itpx:
                    self.data.it[i] += my_stack[i][p[0]][p[1]]
            
            if len(my_itpx) != 0:
                self.data.it /= np.float(len(my_itpx))
            if len(my_i0px) != 0:
                self.data.i0 /= np.float(len(my_i0px))
            
            if len(my_i0px) == 0 or len(my_itpx) == 0:
                self.data.od = np.zeros(len(my_stack))
            else:
                self.data.od = np.log( self.data.i0/self.data.it )
                
        if self.mode == 'linescan':
            n_i0 = 0
            n_it = 0
            self.data.i0 = np.zeros_like(self.data.rawstack[0][0])
            self.data.it = np.zeros_like(self.data.rawstack[0][0])
            
            a = len(self.data.rawstack[0])
            
            for i in range(a):
                if self.data.overlayimg[i][0] == 1.0:
                    n_i0 += 1
                    self.data.i0 += self.data.rawstack[0][i]
                elif self.data.overlayimg[i][0] == -1.0:
                    n_it += 1
                    self.data.it += self.data.rawstack[0][i]
            
            if n_i0 != 0:
                self.data.i0 /= n_i0
            if n_it != 0:
                self.data.it /= n_it
            
            if n_i0 == 0 or n_it == 0:
                self.data.od = np.zeros_like(self.data.rawstack[0][0])
            else:
                self.data.od = np.log( self.data.i0/self.data.it )
        
        self.imgdisplay.redraw(self)
        if self.mode == 'linescan' or self.mode == 'stack':
            self.specdisplay.replotspec()    
    
    def enablectrls(self):
        if self.mode == 'single':
            for btn in [self.autoalignchk, self.imgfirst, self.imgback, self.imgnext, self.imglast, self.ploti0,
                        self.plotit, self.plotod, self.savestack, self.savespectrum, self.savealn, self.saveI0,
                        self.saveall, self.autosavechk]:
                btn.configure(state = tk.DISABLED)
            for btn in [self.odlimit, self.suggestnext, self.addI0, self.addIT,
                        self.clearI0, self.clearIT, self.autosegment, self.showroischk]:
                btn.configure(state = tk.NORMAL)
        
        elif self.mode == 'linescan':
            for btn in [self.suggestnext, self.imgfirst, self.imgback, self.imgnext, self.imglast, self.odlimit,
                        self.autosegment, self.autoalignchk, self.savestack, self.savealn, self.saveall, self.autosavechk]:
                btn.configure(state = tk.DISABLED)
            for btn in [self.addI0, self.addIT, self.clearI0, self.clearIT, self.savespectrum,
                        self.showroischk, self.saveI0, self.ploti0, self.plotit, self.plotod]:
                btn.configure(state = tk.NORMAL)
        
        elif self.mode == 'map':
            for btn in [self.imgback, self.imgnext, self.odlimit, self.addI0, self.addIT, self.autosegment,
                        self.showroischk, self.ploti0, self.plotit, self.plotod, self.autoalignchk,
                        self.saveI0, self.saveall, self.autosavechk, self.clearI0, self.clearIT, self.savespectrum]:
                btn.configure(state = tk.DISABLED)
            for btn in [self.imgfirst, self.imglast, self.savealn, self.savestack, self.suggestnext]:
                btn.configure(state = tk.NORMAL)
        
        elif self.mode == 'stack':
            self.suggestnext.configure(state = tk.DISABLED)
            for btn in [self.autosegment, self.showroischk, self.ploti0, self.plotit, self.plotod, self.savestack,
                        self.savealn, self.saveI0, self.saveall, self.autosavechk, self.imgfirst, self.imglast,
                        self.clearI0, self.clearIT, self.autoalignchk, self.savespectrum, self.imgback, self.imgnext,
                        self.odlimit, self.addI0, self.addIT,]:
                btn.configure(state = tk.NORMAL)
        
        self.stackdisp.set('')
        self.filedisp.set('')
    
    def setfnames(self):
        boxes = [[self.stackfname, '.ncb'], [self.spectrumfname, '.txt'], [self.I0fname, '_i0.txt'], [self.alnfname, '.aln']]
        
        for b in boxes:
            b[0].delete(0, 'end')
        
        if self.mode == 'single':
            tosave = []
        elif self.mode == 'linescan':
            tosave = [1,2]
        elif self.mode == 'map':
            tosave = [0,3]
        elif self.mode == 'stack':
            tosave = range(4)
        
        for k in tosave:
            boxes[k][0].insert(0, path.basename(self.hdrfile)[:-4] + boxes[k][1])
    
    def writeall(self):
        writencb(self)
        writetxt(self)
        writetxt(self, is_i0 = True)
        writealn(self)
        self.filedisp.set('Wrote files to ' + path.dirname(self.hdrfile))
    
    def addxim(self, ximfile):
        self.data.imgfile.append(ximfile)
        self.data.rawstack.append(np.loadtxt(ximfile)[::-1])
        ind = len(self.data.imgfile) - 1
        
        zeropad = self.data.estrlen - len(str(self.data.energies[ind]))
        f = path.basename(ximfile) + '     ' + str(self.data.energies[ind]) + '0'*zeropad + ' eV'
        self.imglabels.append(f)
        
        self.data.shifts = np.append( self.data.shifts, [calculate_shift(self.data.rawstack[0], self.data.rawstack[-1], self.data.imgdims[3])], axis = 0 )
        self.data.alnstack.append( alignoneimage(self.data.rawstack[-1], self.data.shifts[-1]) )
        
        mask = 1 - 1*(self.data.alnstack[-1] == -1)
        self.data.keeppx *= mask
        
        if self.imgselect.get() == self.imglabels[-2]:
            ii = len(self.data.rawstack)-1
        else:
            ii = self.imglabels.index(self.imgselect.get())
        
        self.setxim(ind = ii)
        self.genI0IT()
    
    def timeleft(self):
        tlist = [path.getctime(f) for f in self.data.imgfile]
        for i in range(len(tlist)-1, 0, -1):
            tlist[i] -= tlist[i-1]
        tlist[0] -= self.starttime
        dt = np.mean(np.array(tlist))
        
        return dt
    
    def livestack(self):
        if self.stackrunning:
            ximlist = glob(path.join(path.dirname(self.hdrfile), '*.xim'))
            ximlist = [f for f in ximlist if f not in self.data.imgfile]
            ximlist.sort()
            for f in ximlist:
                self.addxim(f)
                self.chkpoint = time()
            
            my_dt = self.timeleft()
            tleft = my_dt*(len(self.data.energies) - len(self.data.rawstack)) - (time() - self.chkpoint)
            sec = str(int(tleft)%60)
            if len(sec) == 1:
                sec = '0' + sec
            self.stackdisp.set('Stack running... ' + str(int(tleft/60)) + ':' + sec + ' remaining')
            
        if len(self.data.imgfile) == len(self.data.energies):
            self.stackrunning = False
            
            if self.mode == 'stack':
                self.stackdisp.set('Stack complete')
                
                if self.autosave.get() == 1:
                    writencb(self)
                    writealn(self)
                    if self.mode == 'stack':
                        writetxt(self)
                        writetxt(self, is_i0 = True)
            elif self.mode == 'map':
                if self.autoalign.get == 1:
                    self.data.map = genmap(self.data.alnstack[0], self.data.alnstack[1])
                else:
                    self.data.map = genmap(self.data.rawimg[0], self.data.rawimg[1])
                self.stackdisp.set('Map complete')
        
        elif time() - self.chkpoint > 3.0*my_dt:
            self.stackdisp.set('Timed out -- stack aborted or files not found')
            self.stackrunning = False
            
        else:
            self.refresh = Timer(1.0, self.livestack).start()
    
    def setcommands(self):
        # For setting .hdr file
        self.browsebtn.config(command = self.hdrdialog)
        self.recentbtn.config(command = self.mostrecent)
        
        # Choosing ROIs manually
        self.addI0.config(command = lambda: self.addtoroi(1))
        self.addIT.config(command = lambda: self.addtoroi(-1))
        self.clearI0.config(command = lambda: self.clearroi(1))
        self.clearIT.config(command = lambda: self.clearroi(-1))
        self.odlimit.config(command = self.odfilter)
        
        # Image controls
        self.showroischk.config(command = lambda: self.imgdisplay.redraw(self))
        self.autoalignchk.config(command = self.genI0IT)
        self.autosegment.config(command = self.autoselect_rois)
        
        # Image nav buttons
        self.imgfirst.config(command = lambda: self.setxim(ind = 0))
        self.imglast.config(command = lambda: self.setxim(ind = len(self.data.rawstack) - 1))
        self.imgback.config(command = lambda: self.setxim(ind = 'back'))
        self.imgnext.config(command = lambda: self.setxim(ind = 'next'))
        
        # Choose what to plot
        self.ploti0.config(command = self.specdisplay.replotspec)
        self.plotit.config(command = self.specdisplay.replotspec)
        self.plotod.config(command = self.specdisplay.replotspec)
        
        # Write files
        self.savestack.config(command = lambda: writencb(self))
        self.savespectrum.config(command = lambda: writetxt(self))
        self.saveI0.config(command = lambda: writetxt(self, is_i0 = True))
        self.savealn.config(command = lambda: writealn(self))
        self.saveall.config(command = lambda: self.writeall)
        
        
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        
        if system() == 'Windows':
            self.gridpad = 2
            self.btnpad = 0
        else:
            self.gridpad = 2
            self.btnpad = 5
        
        with open('stxmlive_config.txt', 'r') as inifile:
            self.hdrfile = inifile.readline()
            self.workdir = path.dirname(self.hdrfile)
        
        self.data = stxmdata()
        self.mode = 'single'
        self.stackrunning = False
        self.refresh = None
        
        makefilepicker(self)
        makeimgcontrols(self)
        makespectrumcontrols(self)
        makemplpanels(self)
        
        self.pack()
        
        self.showroischk.toggle()
        self.autoalignchk.toggle()
        self.autosavechk.toggle()
        
        self.setcommands()
    
if __name__ == '__main__':
    root = tk.Tk()
    root.wm_title("STXM Live Analysis v1.0")
    app = MainWindow(master = root)
    app.mainloop()
