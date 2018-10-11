
# coding: utf-8

# In[51]:

get_ipython().magic(u'matplotlib inline')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import lab2
import lab3

def otsu_gray(image, hist):
    M=len(image)
    N=len(image[0])
    # 0 background, 1 foreground
    out=np.ones((M,N))
    L=len(hist)
    # escludiamo il primo e ultimo valore
    var_min=255*2557
    T_min=-1
    for T in range(1,L-1):
        mask=image>=T
        out[mask]=0
        wf=np.sum(hist[:T])
        wb=1-wf
        muf=0
        for i in range(0,T):
            muf+=i*hist[i]
        if wf!=0:
            muf=muf/wf
        mub=0
        for i in range(T,L):
            mub+=i*hist[i]
        if wb!=0:
            mub=mub/wb
        sigmaf=0
        if wf!=0:
            for i in range(0,T):
                sigmaf+=(i-muf)*(i-muf)*hist[i]
            sigmaf=sigmaf/wf
        sigmab=0
        if wb!=0:
            for i in range(T,L):
                sigmab+=(i-mub)*(i-mub)*hist[i]
            sigmab=sigmab/wb
            
        varT=wb*sigmab+wf*sigmaf
        if varT < var_min:
            var_min=varT
            T_min=T
    mask=image>=T_min
    out[mask]=255
    print T_min
    return out


def region_growing(image, T):
    
    M=len(image)
    N=len(image[0])
    propagata=-100*np.ones((M+2,N+2))
    propagata[1:M+1,1:N+1]=image
    
    # 0 background, 1 foreground
    S=-1*np.ones((M+2,N+2))
    ivecchia=1;
    jvecchia=1;
    seed=[]
    # ricerca del prossimo pixel non processato (con valore -1)
    for iext in range(ivecchia,M+1):
        for jext in range(jvecchia,N+1):
            if S[iext,jext]==-1:
                seed.append([iext,jext])                
                # si fa qui l'inizializzazione 
                S[iext,jext]=propagata[iext,jext]
                # print "pixel nuovo: " , iext, jext
                # print "valore pixel: ", S[iext, jext]
                # "not seed" => vuol dire che e' vuota, quindi "a" vuol dire che non e' vuota
                while seed:
                    posi,posj=seed.pop()
                    # esploro il vicinato del seed corrente
                    for i in range(posi-1,posi+2):
                        for j in range(posj-1,posj+2):
                            # test di "somiglianza"
                            if S[i,j]==-1 and np.abs(propagata[i,j]-propagata[posi,posj])<T:
                                S[i,j]=S[posi,posj]
                                seed.append([i,j])
    return lab3.crop(S,1,1,M,N)


# In[61]:

col=0
image = cv2.imread('picasso.jpg',col)
segmentata=region_growing(image, 8)
plt.subplot(121), plt.imshow(image,cmap='gray'), plt.title('immagine originale')
plt.subplot(122), plt.imshow(segmentata,cmap='gray'), plt.title('immagine segmenata')



# In[9]:

A=[];
A.append([1,2])
print A
A.append([-1,7])
print A
i,j=A.pop()
print i,j


# In[15]:

a=[]
if a:
    print "ciao"


# In[25]:

print np.abs(-5)


# In[ ]:



