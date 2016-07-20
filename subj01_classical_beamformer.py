# -*- coding: UTF-8 -*-
## (c) Tomasz Steifer 2016
## solutions to http://brain.fuw.edu.pl/edu/index.php/USG/Klasyczna_rekonstrukcja

import numpy as np
import pylab as py

import scipy.signal as ss

f0=5.5e6 # Transducer center frequency [Hz]
fs=50e6 # Sampling frequency [Hz]


c=1490. # Speed of sound [m/s]
pitch=0.00021 #Transducer's pitch
NT=192 # Number of full aperture elements
Nrec=64 #Number of receive events/lines
Ntr=64 # Transmit subaperture
Rf1 = 40*(1e-3) #depth of focal point

fname="usg1_bf_nitki.npy" #numpy file with RF data from Ultrasound Research Platform

RF=np.load(fname)
Mz=np.shape(RF)[1]
##reminder about data:
##first dimension -> receive transducers //Nrec
##second dimension -> time of acquisition ~ depth
##third dimension -> transmit event  // (NT - Ntr)

#First, we need to generate transmit delays - to be used in reconstruction

transmit_delays = -(np.arange(Ntr)-(float(Ntr)/2-0.5))*pitch
transmit_delays = (np.sqrt(transmit_delays**2+Rf1**2)-Rf1)/c*fs

#For sake of clarity, beamforming is performed in loops
image=np.zeros((NT-Ntr,Mz)) #
tmp=np.zeros((Mz,Ntr))
for line in range(NT-Ntr):
    for k in range(Nrec):
        tmp0=RF[line+k,transmit_delays[k]:Mz+transmit_delays[k],line]
        tmp[:len(tmp0),k]=tmp0
    tmp2=np.mean(tmp,axis=1)
    image[line,:len(tmp2)]=tmp2

#Some simple filters
b, a = ss.butter(10, 0.05, 'highpass')    
image = ss.filtfilt(b, a, image,axis=1)
b, a = ss.butter(10, 0.7, 'lowpass')    
image = ss.filtfilt(b, a, image,axis=1)


#We need to interpolate the image
xx=np.linspace(-(NT - Ntr)/2.*pitch,((NT - Ntr))/2.*pitch,(NT - Ntr)) #old x scale
zz=np.linspace(0,np.shape(image)[1]/2*(1./fs)*c,np.shape(image)[1]) #old depth scale
from scipy import interpolate
#f=interpolate.interp2d(xx,zz,image,kind='cubic')
f=interpolate.interp2d(xx,zz,image.transpose(),'cubic')

#new (natural) grid
step=c/(2*f0)
step=step/4
maxdepth=np.max(zz)
mindepth=0.005 #try to set minimal depth to 0 and see what happens
maxWidth=0.01
minWidth=0.01

X0 = np.arange(-minWidth,maxWidth,step)#.reshape((-1,1))
R0 = np.arange(mindepth,maxdepth,step)

#and interpolation
image=f(X0,R0)

#Lastly, envelope detection using Hilbert transform
image=np.abs(ss.hilbert(image,axis=0))

low=50 #lower bound for dB scale

#and log-conversion etc <- this may also be done by appropriate matplotlib methods
indices = image>0
indices2 =image<0
indices = indices+indices2
image=np.abs(image)/np.max(np.abs(image[indices]))
image[indices]=10*np.log(image[indices])

indices = indices <-low
image[indices]=-low
indices = image>=0
image[indices]=0

#Let's have a look:
py.imshow(image,cmap="gray")
py.show()
