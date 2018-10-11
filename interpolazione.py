
# coding: utf-8

# In[285]:

# interpolazione
get_ipython().magic(u'matplotlib inline')
import cv2
import numpy as np
import matplotlib.pyplot as plt


def interp_copia(image):
    size=image.shape
    m=len(image)
    n=len(image[0])
    # dimensione nuova immagine è il doppio di quella di input
    M=2*m
    N=2*n
    # numero di canali
    ch=3
    # trucco per immagini in scala grigi
    if len(size)==2:
        ch=1
        im_grigia=np.zeros((m,n,ch),dtype=np.uint8)
        im_grigia[:,:,0]=image
        image=im_grigia
    im_int=np.zeros((M,N,ch),dtype=np.uint8)
    # le righe e colonne pari sono uguali all'immagine di input
    im_int[::2,::2,:]=image
    # le colonne dispari delle righe pari assumono il valore della colonna precedente
    im_int[::2,1::2,:]=im_int[::2,0::2,:]
    # le righe dispari assumono il valore della riga precedente
    im_int[1::2,:,:]=im_int[::2,:,:]
    # trucco per immagini in scala grigi
    if ch==1:
        im_int_grigia=np.zeros((M,N))
        im_int_grigia=im_int[:,:,0]
        im_int=im_int_grigia
    return im_int


def interp_media(image):
    size=image.shape
    m=len(image)
    n=len(image[0])
    # dimensione nuova immagine è il doppio di quella di input
    M=2*m
    N=2*n
    # numero di canali
    ch=3
    # trucco per immagini in scala grigi
    if len(size)==2:
        ch=1
        im_grigia=np.zeros((m,n,ch),dtype=np.uint8)
        im_grigia[:,:,0]=image
        image=im_grigia
    im_int2=np.zeros((M,N,3),dtype=np.uint8)
    # le righe e colonne pari sono uguali all'immagine di input
    im_int2[::2,::2,:]=image
    # le colonne dispari j delle righe pari assumono la media delle colonne j-1 e j+1
    # attenzione  fare il *0.5 per ognuno, e non dopo, se no va subito in overflow e mena
    im_int2[::2,1:N-3:2,:]=(0.5*im_int2[::2,:N-4:2,:]+0.5*im_int2[::2,2:N-2:2,:])
    # ultima colonna delle righe pari assume il valore della colonna precedente
    im_int2[::2,N-1,:]=im_int2[::2,N-2,:]
    # le righe dispari i assumono il valore medio delle righe i-1 e i+1
    im_int2[1:M-3:2,:,:]=(0.5*im_int2[:M-4:2,:,:]+0.5*im_int2[2:M-2:2,:,:])
    # ultima riga assume il valore della penultima
    im_int2[M-1,:,:]=im_int2[M-2,:,:]
    # trucco per immagini in scala grigi
    if col==0:
        im_int_grigia=np.zeros((M,N))
        im_int_grigia=im_int2[:,:,0]
        im_int2=im_int_grigia
    return im_int2


# In[289]:

col=1
image = cv2.imread('mont.jpg',col)
if col==1:
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image,cmap='gray')
print image.shape


# In[290]:

# prima interpolazone
im_int=interp_copia(image)
plt.imshow(im_int,cmap='gray')


# In[291]:

# seconda interpolazione
im_int2=interp_media(image)
plt.imshow(im_int2,cmap='gray')


# In[292]:

if col==1:
    image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    im_int=cv2.cvtColor(im_int, cv2.COLOR_RGB2BGR)
    im_int2=cv2.cvtColor(im_int2, cv2.COLOR_RGB2BGR)
cv2.imwrite('be_int.jpg',im_int)
cv2.imwrite('be_int2.jpg',im_int2)
cv2.imwrite('be_g.jpg',image)


# In[ ]:




# In[ ]:




# In[ ]:



