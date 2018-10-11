
# coding: utf-8

# In[56]:

get_ipython().magic(u'matplotlib inline')
import lab1
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[57]:

def convoluzione(image, kernel):
    size=image.shape
    M=len(image)
    N=len(image[0])
    K=len(kernel)
    L=len(kernel[0])
    col=3
    # caso grigio (solito trucco)
    if(len(size)==2):
        col=1
        im_grigia=np.zeros((M,N,col))
        im_grigia[:,:,0]=image
        image=im_grigia
    # dimensione output è maggiore
    output=np.zeros((M+K-1,N+L-1,col))
    # convoluzione con la formula "efficiente"
    for i in range(0,M):
        for k in range(0,K):
            for j in range(0,N):
                for l in range(0,L):
                    output[i+k,j+l,:]+=image[i,j]*kernel[k,l]
    # se l'input era grigio restituisco un'immagine di tipo grayscale
    if(len(size)==2):
        out_grig=np.zeros((M+K-1,N+L-1))
        out_grig=output[:,:,0]
        output=out_grig
    return output.astype(np.uint8)


def crop(image,n1,n2,N1,N2):
    # (n1,n2) è la posizione del pixel in alto a sx da cui parte il crop
    # (N1,N2) è la dimensione del crop in output
    M=len(image)
    N=len(image[0])
    if (n1+N1 > M or n2+N2 > N):
        print "errore dimensioni"
        return None
    crop=image[n1:n1+N1,n2:n2+N2]
    return crop


def filtraggio(image, kernel, pad):
    # nota: si suppone che il kernel abbia sempre dimensioni dispari (n_righe e n_colonne)
    M=len(image)
    N=len(image[0])
    K=len(kernel)
    L=len(kernel[0])
    if(pad == ""):
        # filtraggio
        im_filtr=convoluzione(image, kernel)
        # posizione del pixel da cui iniziare il cropping (vedi slide 18 del filtraggio)
        priga=(K-1)/2
        pcolonna=(L-1)/2
    elif(pad == "prop"):
        im_prop=propagation(image,[(K-1)/2,(L-1)/2])
        # filtraggio
        im_filtr=convoluzione(im_prop, kernel)
        # posizione del pixel da cui iniziare il cropping
        priga=K-1
        pcolonna=L-1
    else:
        im_mir=mirroring(image,[(K-1)/2,(L-1)/2])
        # filtraggio
        im_filtr=convoluzione(im_mir, kernel)
        # posizione del pixel da cui iniziare il cropping
        priga=K-1
        pcolonna=L-1
    # faccio il crop con posizione di partenza (priga,pcolonna) e con dimensione pari all'immagine di input
    return crop(im_filtr, priga, pcolonna, M, N)
        

def propagation(image, dim_bordi):
    # dimBordi è un vettore con due elementi: il numero di colonne da aggiungere a sx e a dx dell'immagine
    # e il numero di righe da aggiungere in alto e in basso
    r=dim_bordi[0]
    c=dim_bordi[1]
    M=len(image)
    N=len(image[0])
    size=image.shape
    col=3
    # caso grigio
    if(len(size)==2):
        col=1
        im_grigia=np.zeros((M,N,col),dtype=np.uint8)
        im_grigia[:,:,0]=image
        image=im_grigia
    im_prop=np.zeros((M+2*r,N+2*c,col),dtype=np.uint8)
    # parte centrale è uguale all'immagine di input
    im_prop[r:r+M,c:c+N]=image
    # bordi laterali
    for i in range(0,c):
        im_prop[r:r+M,i]=image[:,0]
        im_prop[r:r+M,i+c+N]=image[:,-1]
    # bordi sopra sotto
    for i in range(0,r):
        im_prop[i,c:c+N]=image[0,:]
        im_prop[i+r+M,c:c+N]=image[-1,:]
    # angoli
    for i in range(0,r):
        for j in range(0,c):
            im_prop[i,j]=image[0,0]
            im_prop[r+M+i,j]=image[-1,0]
            im_prop[i,c+N+j]=image[0,-1]
            im_prop[r+M+i,c+N+j]=image[-1,-1]
    # caso grigio
    if(len(size)==2):
        prop_grig=np.zeros((M,N))
        prop_grig=im_prop[:,:,0]
        im_prop=prop_grig
    return im_prop


def mirroring(image, dim_bordi):
    # dimBordi è un vettore con due elementi: il numero di colonne da aggiungere a sx e a dx dell'immagine
    # e il numero di righe da aggiungere in alto e in basso
    r=dim_bordi[0]
    c=dim_bordi[1]
    M=len(image)
    N=len(image[0])
    size=image.shape
    col=3
    # caso grigio
    if(len(size)==2):
        col=1
        im_grigia=np.zeros((M,N,col),dtype=np.uint8)
        im_grigia[:,:,0]=image
        image=im_grigia
    im_prop=np.zeros((M+2*r,N+2*c,col),dtype=np.uint8)
    # parte centrale è uguale all'immagine di input
    im_prop[r:r+M,c:c+N]=image
    # bordi laterali
    im_prop[r:r+M,:c]=image[:,N-c:]
    im_prop[r:r+M,c+N:]=image[:,:c]
    # bordi sopra sotto
    im_prop[:r,c:c+N]=image[M-r:,:]
    im_prop[r+M:,c:c+N]=image[:r,:]
    # angoli
    im_prop[:r,:c]=image[M-r:,N-c:]
    im_prop[r+M:,:c]=image[:r,N-c:]
    im_prop[:r,c+N:]=image[M-r:,:c]
    im_prop[r+M:,c+N:]=image[:r,:c]
    
    # caso grigio
    if(len(size)==2):
        prop_grig=np.zeros((M,N))
        prop_grig=im_prop[:,:,0]
        im_prop=prop_grig
    return im_prop

def median_filter(image, K):
    # K è la dimensione del lato del "neighborhood" utilizzato per applicare il filtro mediano (che si suppone
    # essere quadrato), e tale che K^2 sia dispari, es K=3
    if((K*K)%2 ==0 ):
        return None
    M=len(image)
    N=len(image[0])
    size=image.shape
    col=3
    # caso grigio
    if(len(size)==2):
        col=1
        im_grigia=np.zeros((M,N,col),dtype=np.uint8)
        im_grigia[:,:,0]=image
        image=im_grigia
    im_filtr=np.zeros((M,N,col),dtype=np.uint8)
    # semiK sarebbe l'equivalente del K della slide 18 del filtraggio
    semiK=(K-1)/2
    im_prop=propagation(image, [semiK, semiK])
    for i in range(semiK,im_prop.shape[0]-semiK):
        for j in range(semiK,im_prop.shape[1]-semiK):
            pixels=im_prop[i-semiK:i+semiK+1,j-semiK:j+semiK+1]
            # ordinamento in due passi: prima per righe (axis=0) e poi per colonne (axis=1)
            pixels=np.sort(pixels, axis=0)
            pixels=np.sort(pixels, axis=1)
            im_filtr[i-semiK,j-semiK]=pixels[semiK,semiK]
    # caso grigio
    if(len(size)==2):
        filtr_grig=np.zeros((M,N))
        filtr_grig=im_filtr[:,:,0]
        im_filtr=filtr_grig
    return im_filtr
    


# In[58]:

# definizione filtro

## filtro media
filtro=(1/9.0)*np.ones((3,3))

## filtro per riduzione intensità
#filtro=np.zeros((3,3))
#filtro[0,0]=0.5

# caricamento immagine
col=1
image = cv2.imread('mont.jpg',col)
if(col==1):
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# In[59]:

# filtraggio
im_filtr=convoluzione(image,filtro)

plt.subplot(121), plt.imshow(image,cmap='gray'), plt.title('immagine originale')
plt.subplot(122), plt.imshow(im_filtr,cmap='gray'), plt.title('immagine filtrata')
print image.shape
print im_filtr.shape


# In[60]:

# filtraggio con cropping e propagazione
im_filtr2=filtraggio(image,filtro,"prop")
plt.subplot(121), plt.imshow(image,cmap='gray'), plt.title('immagine originale')
plt.subplot(122), plt.imshow(im_filtr2,cmap='gray'), plt.title('immagine filtrata')


# In[61]:

# filtraggio con cropping e mirroring
im_filtr3=filtraggio(image,filtro,"mirr")
plt.subplot(121), plt.imshow(image,cmap='gray'), plt.title('immagine originale')
plt.subplot(122), plt.imshow(im_filtr2,cmap='gray'), plt.title('immagine filtrata')


# In[62]:

# filtraggio con mediana (solo nel caso grigio, nel caso a colori è difficile definire l'ordinamento)
sporcata=lab1.salt_pepper(image,0.05)
mediana=median_filter(sporcata, 3)
plt.subplot(121), plt.imshow(sporcata,cmap='gray'), plt.title('immagine originale')
plt.subplot(122), plt.imshow(mediana,cmap='gray'), plt.title('immagine filtrata')


# In[63]:

print image.shape
print sporcata.shape
print mediana.shape
print im_filtr2.shape
print im_filtr3.shape


# scrittura delle immagini
if col==1:
    im_filtr=cv2.cvtColor(im_filtr, cv2.COLOR_RGB2BGR)
    im_filtr2=cv2.cvtColor(im_filtr2, cv2.COLOR_RGB2BGR)
    im_filtr3=cv2.cvtColor(im_filtr3, cv2.COLOR_RGB2BGR)
    sporcata=cv2.cvtColor(sporcata, cv2.COLOR_RGB2BGR)
    mediana=cv2.cvtColor(mediana, cv2.COLOR_RGB2BGR)
cv2.imwrite('mont_filtr.jpg',im_filtr)
cv2.imwrite('mont_filtr2.jpg',im_filtr2)
cv2.imwrite('mont_filtr3.jpg',im_filtr3)
cv2.imwrite('mont_pepper.jpg',sporcata)
cv2.imwrite('mont_median.jpg',mediana)


# In[ ]:



