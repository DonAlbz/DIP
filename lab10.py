
# coding: utf-8

# In[32]:

get_ipython().magic(u'matplotlib inline')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import funzioni

# precondizione: l'immagine deve essere con numero di righe e colonne pari
def frequency_filtering(image, filtro):
    M = len(image)
    N = len(image[0])
    H = len(filtro)
    K = len(filtro[0])
    # 0 padding
    padded_image = padding(image, M + H - 1, N + K -1)
    # (-1)*(x+y)
    # mi fermo a M e N perchè tanto poi sono zeri
    for i in range(0,M):
        for j in range(0,N):
            x = j - (N-1)/2
            y = (M-1)/2 - i
            padded_image[i,j] = padded_image[i,j] * np.power(-1,x+y)
    Image_F = np.fft.fft2(padded_image)
    for u in range(0,H):
        for v in range(0,K):
            x = v - (K-1)/2
            y = (H-1)/2 - u
            filtro[u,v] = filtro[u,v] * np.power(-1,x+y)
    filtro_spazio = np.fft.ifft2(filtro)
    padded_filtro = padding(filtro_spazio, M + H - 1, N + K -1)
    
    filtro = np.fft.fft2(padded_filtro)
    output_freq = np.multiply(filtro,Image_F)
    output = np.real(np.fft.ifft2(output_freq))
    for i in range(0,len(output)):
        for j in range(0,len(output[0])):
            x = j - (len(output[0])-1)/2
            y = (len(output)-1)/2 - i
            output[i,j] = output[i,j] * np.power(-1,x+y)
    return funzioni.crop(output, H/2, K/2, M,N)



# precondizione: l'immagine deve essere con numero di righe e colonne pari
def correlazione(image, image2):
    M = len(image)
    N = len(image[0])
    H = len(image2)
    K = len(image2[0])
    # 0 padding
    padded_image = padding(image, M + H - 1, N + K -1)
    Image_F = np.fft.fft2(padded_image)
    padded_image2 = padding(image2, M + H - 1, N + K -1)
    Image2_F = np.fft.fft2(padded_image2)
    Image2_F_coniug = np.conj(Image2_F)
    output_freq = np.multiply(Image_F, Image2_F_coniug)
    output = np.real(np.fft.ifft2(output_freq))
    return funzioni.crop(output, H/2, K/2, M,N)
    

    
def padding(image, Mtot, Ntot):
    M = len(image)
    N = len(image[0])
    out = np.zeros((Mtot, Ntot))
    out[0:M, 0:N] = image
    return out 
    
    
    
    
    


# In[33]:

col=0
image = cv2.imread('letters.jpg',col)
if(col==1):
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
M = len(image)
N = len(image[0])
if(M%2 == 1 and N%2 == 1):
    image=image[0:M-1,0:N-1]
elif(M%2 == 1):
    image=image[0:M-1,0:N]
elif(N%2 == 1):
    image=image[0:M,0:N-1]


# In[25]:



dim_banda = 50
basso = np.zeros((len(image),len(image[0])))
basso[len(image)/2-dim_banda:len(image)/2+dim_banda, len(image[0])/2-dim_banda:len(image[0])/2+dim_banda] = 1

alto = np.ones((len(image),len(image[0])))
alto[len(image)/2-dim_banda:len(image)/2+dim_banda, len(image[0])/2-dim_banda:len(image[0])/2+dim_banda] = 0

output = frequency_filtering(image, alto)
plt.subplot(121), plt.imshow(image,cmap='gray'), plt.title('immagine originale')
plt.subplot(122), plt.imshow(output,cmap='gray'), plt.title('immagine filtrata')



# In[36]:

image2 = cv2.imread('k.jpg',col)
corr = correlazione(image, image2)
plt.imshow(corr,cmap='gray')


# In[27]:

if col==1:
    output=cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

cv2.imwrite('corr.jpg',corr)
cv2.imwrite('mont_grigio.jpg',image)


# In[60]:

A=np.ones((1,1))
B=padding(A, 5,5)
print B


# In[17]:

print -1^2


# In[69]:

print np.multiply(1j, 3-2j)


# In[ ]:

if(M%2 == 0 and N%2 == 0):
    image=image[0:M-1,0:N-1]
elif(M%2 == 0):
    image=image[0:M-1,0:N]
elif(N%2 == 0):
    image=image[0:M,0:N-1]
M = len(image)
N = len(image[0])

