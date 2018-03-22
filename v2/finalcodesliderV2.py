
# coding: utf-8

# In[1]:

#plots both individual and looped for all transverse aberations 


# In[34]:

import numpy as np
from matplotlib import pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
from pylab import scatter
pi=np.pi

# In[98]:

#fourth order astigmatism ro=field, ra=aperture
#A,B,C,D,E 0 or +-1 to select which terms to use
def circle(x,y):
    #Takes a input of X and Y as meshgrid matrices and returns the values which are less than magnitude one, to select a circle.
    r= np.sqrt(np.square(x)+np.square(y))
    temp = r<1
    return temp.astype(float)

def astigmatism(circPnts,obj,ap,(null,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16)):
    #cirPnts is the output of circle which will be used to window the grid of points in aperture space to a circle.
    #obj is an nxn matrix of field locations, and ap is an mxm matrix of aperture locations.
    # the following input vector is used to turn on/off certain aberration terms.
    
    apSize=np.size(ap[:,0])
    obSize=np.size(obj[:,0])
    circSize=np.size(circPnts)
    
    #this reshapes the inputs and generates every ray which will gothrough system.
    rox=np.repeat(obj[:,0],apSize).reshape(obSize,apSize)
    roy=np.repeat(obj[:,1],apSize).reshape(obSize,apSize)
    
    rax=np.repeat(ap[:,0],obSize).reshape(apSize,obSize).T
    ray=np.repeat(ap[:,1],obSize).reshape(apSize,obSize).T
    
    #this generates a set of 'ideal' image points with whcih to reference all the aberrated points
    shift=np.repeat(obj,circSize,axis=0)
    
    #the invariants
    a=rox**2
    b=roy**2
    c=rax**2
    d=ray**2
    g=np.array([np.dot(ap,i) for i in obj])
    h=np.array([np.dot(ap,i) for i in obj])
    j=np.repeat(np.einsum('ij,ij->i',obj,obj),apSize).reshape(obSize,apSize)
    k=(np.repeat(np.einsum('ij,ij->i',ap,ap),obSize).reshape(apSize,obSize)).T
    null=0
    
    #group 2 Astigmatism and field curvature
    out=C1*g**2+C2*g*k+C3*k**2+C4*j*k+C5*h*j

    #taking the gradient effecteively generating the transverese aberrations
    outReshape=out.reshape(obSize,int(np.sqrt(apSize)),int(np.sqrt(apSize)))
    grad=np.array([np.gradient(i.T) for i in outReshape])
    grad=np.array(grad.reshape(obSize,2,apSize))
    
    #seperating the gradient into x and y and selecting only the points whch are within our cirucluar aperture
    gradx=np.array([np.take(grad[i][0],circPnts) for i in range(0,obSize)]).reshape(obSize*circSize)#use take and the ap mask in this loop
    grady=np.array([np.take(grad[i][1],circPnts) for i in range(0,obSize)]).reshape(obSize*circSize)#use take and the ap mask in this loop
    

    #Shifting the transverse aberrations by their ideal image location
    gx=np.array(gradx)+shift[:,0]
    gy=np.array(grady)+shift[:,1]
    
    return np.array([gx,gy])
# In[104]:

def points(pnts,fPnts):

    #build the circular field
    x=np.linspace(-1.3,1.3,num=pnts)
    X,Y=np.meshgrid(x,x)

    circ=circle(X,Y)
    pntLoc=(np.where(circ.flatten()==1))

    
    fx=np.linspace(-1,1,num=fPnts)
    fX,fY=np.meshgrid(fx,fx)
    
    apVect=np.hstack((X.reshape(pnts**2,1),X.T.reshape(pnts**2,1)))
    field=np.hstack((fX.reshape((fPnts)**2,1),fX.T.reshape((fPnts)**2,1)))
    return np.array([apVect,field,pntLoc])

coef1=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

pointField=points(30,15)


wvfrm=astigmatism(pointField[2],pointField[1],pointField[0],coef1)

fig = plot.figure()
plt_ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.25, bottom=0.4)
axis_color = 'lightgoldenrodyellow'

plot.axes(plt_ax)
scat, = plot.plot(wvfrm[0],wvfrm[1],linestyle='None',marker=',',alpha=.5)
plot.axis('equal')


# Define an axes area and draw a slider in it
coef_slider_ax  = fig.add_axes([0.25, 0.35, 0.65, 0.03],axisbg=axis_color)
coef_slider = Slider(coef_slider_ax, 'Astigmatism', -.3, .3, valinit=0)

coef2_slider_ax  = fig.add_axes([0.25, 0.3, 0.65, 0.03],axisbg=axis_color)
coef2_slider = Slider(coef2_slider_ax, 'Coma', -.3, .3, valinit=0)

coef3_slider_ax  = fig.add_axes([0.25, 0.25, 0.65, 0.03],axisbg=axis_color)
coef3_slider = Slider(coef3_slider_ax, 'Spherical Aberration', -.3, .3, valinit=0)

coef4_slider_ax  = fig.add_axes([0.25, 0.2, 0.65, 0.03],axisbg=axis_color)
coef4_slider = Slider(coef4_slider_ax, 'Field Curvature', -.3, .3, valinit=0)

coef5_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03],axisbg=axis_color)
coef5_slider = Slider(coef5_slider_ax, 'Distortion', -1, 1, valinit=0)

pnt_slider_ax  = fig.add_axes([0.25, 0.1, 0.65, 0.03],axisbg=axis_color)
pnt_slider = Slider(pnt_slider_ax, 'Aperture Points', 5, 31, valinit=18)

fpnt_slider_ax  = fig.add_axes([0.25, 0.05, 0.65, 0.03],axisbg=axis_color)
fpnt_slider = Slider(fpnt_slider_ax, 'Field Points', 5, 31, valinit=18)

# Define an action for modifying the line when any slider's value changes
def update(val):
    coef1=coef_slider.val
    coef2=coef2_slider.val
    coef3=coef3_slider.val
    coef4=coef4_slider.val
    coef5=coef5_slider.val
    pointFieldNew=points(int(pnt_slider.val),int(fpnt_slider.val))
    
    xVals=astigmatism(pointFieldNew[2],pointFieldNew[1],pointFieldNew[0],np.array([0,coef1,coef2,coef3,coef4,coef5,0,0,0,0,0,0,0,0,0,0,0]))[0]
    yVals=astigmatism(pointFieldNew[2],pointFieldNew[1],pointFieldNew[0],np.array([0,coef1,coef2,coef3,coef4,coef5,0,0,0,0,0,0,0,0,0,0,0]))[1]
    
    #vals=np.vstack(xVals,yVals)
    scat.set_xdata(xVals)
    scat.set_ydata(yVals)
    
    fig.canvas.draw_idle()
coef_slider.on_changed(update)
coef2_slider.on_changed(update)
coef3_slider.on_changed(update)
coef4_slider.on_changed(update)
coef5_slider.on_changed(update)
pnt_slider.on_changed(update)
fpnt_slider.on_changed(update)


plot.show()