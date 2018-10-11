
# coding: utf-8

# In[118]:

# gli istogrammi sono utili anche per confrontare immagini diverse ma con contenuto simile,
# per assurdo anche due immagini tali che una è il ribaltamento dell'altra hanno lo stesso istogramma
# insomma l'istogramma è una feature utile

import lab1
get_ipython().magic(u'matplotlib inline')
import cv2
import numpy as np
import matplotlib.pyplot as plt


def hist_cum(image, L):
    size=image.shape
    m=len(image)
    n=len(image[0])
    h=[]
    F=[]
    ch=1
    if len(size)==2:
        h=np.zeros((ch,L))
        im_grigia=np.zeros((m,n,1),dtype=np.uint8)
        im_grigia[:,:,0]=image
        image=im_grigia
        
    else:
        ch=3
        h=np.zeros((ch,L))
    # calcolo istogramma
    for k in range(0,ch):
        print "color:" , k
        for i in range(0,m):
            for j in range(0,n):
                h[k,image[i,j,k]]=h[k,image[i,j,k]]+1
    dim=m*n
    h=h/dim
    # calcolo cumulativa
    F=np.zeros(h.shape)
    F[:,0]=h[:,0]
    for i in range(1,L):
        F[:,i]=F[:,i-1]+h[:,i]
    mat=np.zeros((2,ch,L))
    mat[0,:,:]=h
    mat[1,:,:]=F
    return mat
        
def hist_eq(image, F):
    size=image.shape
    m=len(image)
    n=len(image[0])
    L=len(F[0])
    ch=3
    if len(size)==2:
        ch=1
        im_grigia=np.zeros((m,n,1),dtype=np.uint8)
        im_grigia[:,:,0]=image
        image=im_grigia
    Ieq=np.zeros((m,n,ch),dtype=np.uint8)
    for k in range(0,ch):
        for i in range(0,m):
            for j in range(0,n):
                Ieq[i,j,k]=(L-1)*F[k,image[i,j,k]]
    if(ch==1):
        Ieq_grigia=np.zeros((m,n))
        Ieq_grigia=Ieq[:,:,0]
        Ieq=Ieq_grigia
    return Ieq



# In[122]:

col=1
image = cv2.imread('2.jpg',col)
if(col==1):
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
colori=['r','b','g']
mat=hist_cum(image,256)
for i in range(0,col*3+1-col):
    plt.plot(mat[0,i,:],colori[i])
plt.xlim([0,256])
plt.title('istogramma')


# In[123]:

im_eq=hist_eq(image,mat[1,:,:])
plt.imshow(im_eq,cmap='gray')
plt.subplot(121), plt.imshow(image,cmap='gray'), plt.title('immagine con poco contrasto')
plt.subplot(122), plt.imshow(im_eq,cmap='gray'), plt.title('immagine con aumento contrasto')


# In[124]:

# istogramma dell'immagine equalizzata
mat=hist_cum(im_eq,256)
for i in range(0,col*3+1-col):
    plt.plot(mat[0,i,:],colori[i])
plt.xlim([0,256])
plt.title('istogramma')


# In[ ]:



