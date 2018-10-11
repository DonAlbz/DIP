
# coding: utf-8

# In[1]:

#get_ipython().magic(u'matplotlib inline')
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('img1.jpg',1)
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)


# In[4]:

## funzione per la trasposizione
def trasponi(image):
    size=image.shape
    if len(size)==2:
        m=len(image)
        n=len(image[0])
        trasp=np.zeros((n,m))
        for i in range(0,m):
            for j in range(0,n):
                trasp[j][i]=image[i][j]
        return trasp
    else:
        m=len(image)
        n=len(image[0])
        k=len(image[0][0])
        trasp_col=np.zeros((n,m,3),dtype=np.uint8)
        for i in range(0,m):
            for j in range (0,n):
                trasp_col[j][i]=image[i][j]
        return trasp_col

im = cv2.imread('img1.jpg',1)
# questo comando serve per rimettere i colori come RGB da BGR, deve essere fatto solo
# se l'immagine è importata a colori
im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
trasp=trasponi(im)
plt.imshow(trasp,cmap='gray')
plt.show()

# In[5]:

## funzione per flip verticale
def vert_flip(image):
    flipped=np.zeros(image.shape,dtype=np.uint8)
    m=len(image)
    n=len(image[0])
    for i in range(0,m):
        for j in range(0,n):
            flipped[i][j]=image[i][n-1-j]
    return flipped

im = cv2.imread('img1.jpg',1)
# questo comando serve per rimettere i colori come RGB da BGR, deve essere fatto solo
# se l'immagine è importata a colori
im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
flipped=vert_flip(im)
plt.imshow(flipped,cmap='gray')
plt.show()

# In[6]:

## funzione per crop
def crop(image,n1,n2,N1,N2):
    M=len(image)
    N=len(image[0])
    size=image.shape
    if len(size)==2:
        crop=np.zeros((N1,N2),dtype=np.uint8)
        k=0
        while k < N1 and k+n1 < M:
            l=0
            while l < N2 and l+n2 < N:
                crop[k][l]=image[n1+k][n2+l]
                l=l+1
            k=k+1
        return crop
    else:
        crop=np.zeros((N1,N2,3),dtype=np.uint8)
        k=0
        while k < N1 and k+n1 < M:
            l=0
            while l < N2 and l+n2 < N:
                crop[k][l]=image[n1+k][n2+l]
                l=l+1
            k=k+1
        return crop
    
im = cv2.imread('img1.jpg',1)
# questo comando serve per rimettere i colori come RGB da BGR, deve essere fatto solo
# se l'immagine è importata a colori
im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
cropped=crop(im,50,100,200,300)
plt.imshow(cropped,cmap='gray')



# In[8]:

## funzione per negativo
def negativo(image):
    size=image.shape
    m=len(image)
    n=len(image[0])
    # se size ha 3 elementi vuol dire che era a colori
    if len(size)==3:
        neg=np.zeros((m,n,3),dtype=np.uint8)
        for i in range(0,m):
            for j in range(0,n):
                for k in range(0,3):
                    neg[i][j][k]=255-image[i][j][k]
        return neg
    else:
        
        neg=np.zeros((m,n),dtype=np.uint8)
        for i in range(0,m):
            for j in range(0,n):
                neg[i][j]=255-image[i][j]
        return neg
im = cv2.imread('img1.jpg',1)
# questo comando serve per rimettere i colori come RGB da BGR, deve essere fatto solo
# se l'immagine è importata a colori
im=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
neg=negativo(im)
plt.imshow(neg,cmap='gray') 
    


# In[10]:

## aggiungere rumore
im = cv2.imread('img1.jpg',0)
m=len(image)
n=len(image[0])
for i in range(0,m):
    rum=np.random.normal(0,50,n)
    for j in range(0,n):
        a=im[i][j]+rum[j]
        im[i][j]=a
im=np.floor(im)
plt.imshow(im,cmap='gray') 


# In[11]:

## salt pepper
def salt_pepper(image,soglia):
    m=len(image)
    n=len(image[0])
    size=image.shape
    if len(size)==2:
        for i in range(0,m):
            for j in range(0,n):
                p=np.random.rand()
                if p < soglia:
                    image[i][j]=0
                elif p > (1-soglia):
                    image[i][j]=255
        return image
    else:
        for i in range(0,m):
            for j in range(0,n):
                p=np.random.rand()
                if p < soglia:
                    image[i][j][:]=0
                elif p > (1-soglia):
                    image[i][j][:]=255
        return image
img = cv2.imread('img1.jpg',1)
img2=salt_pepper(img,0.05)
plt.imshow(img2,cmap='gray')

# In[12]:

# contrast sensitivity function
N=800
x=np.linspace(0,5,N)
y=np.linspace(0,1,N)
I=np.zeros((N,N),dtype=np.uint8)
for i in range(0,N):
    for j in range(0,N):
        I[i][j]=128*(1+np.sin(np.exp(x[j]))*np.power(y[i],3))
plt.imshow(I,cmap='gray') 
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:



