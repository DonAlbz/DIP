
# coding: utf-8

# In[51]:

get_ipython().magic(u'matplotlib inline')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import funzioni

def erosione(image, kernel):
    # esempi di kernel:
    # N8=[[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]]
    # N4=[[0,-1],[-1,0],[1,0],[0,1]]
    # kernel contiene gli indici relativi rispetto al pixel centrale (es: [-1,-1],[1,-1],etc..)
    M=len(image)
    N=len(image[0])
    out=np.zeros((M,N))
    # lavoriamo da 1 a M-1 e gestiamo il bordo con "politica propagazione background", cioe' mettiamo a background il bordo
    for i in range(1,M-1):
        for j in range(1,N-1):
            # lavoro sui pixel del foreground (=255)
            if image[i,j]==255:
                # applico l'opreazione morfologica
                # trovato e' una variabile che mi dice se trovo pixel non foreground nel controllo morfologico
                trovato=False
                for l in kernel:
                    i_k,j_k=l
                    if image[i+i_k,j+j_k]==0:
                        trovato=True                        
                        break
                # se tutti i pixel rispettavano i
                if not trovato:
                    out[i,j]=255
    out[0,:]=0
    out[:,0]=0
    out[-1,:]=0
    out[:,-1]=0
    return out


def dilatazione(image, kernel):
    M=len(image)
    N=len(image[0])
    out=np.zeros((M,N))
    mask=image==0
    out[mask]=255
    out=erosione(out,kernel)
    output=np.zeros((M,N))
    mask=out==0
    output[mask]=255
    return output


def opening(image,kernel):
    # allarga i buchi
    return dilatazione(erosione(image,kernel),kernel)
    

def closing(image,kernel):
    # chiude i buchi
    return erosione(dilatazione(image,kernel),kernel)
        

def contorni(image,kernel):
    erosa=erosione(image,kernel)
    return image-erosa
    
def scheletro(image):
    # il kernel e' sempre 3x3
    kernel=[[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]]    
    M=len(image)
    N=len(image[0])
   
    # candidati e' una lista che contine i pixel candidati ad essere erosi
    candidati=[]
    for i in range(1,M-1):
        for j in range(1,N-1):
            if image[i,j]==255:
                contatore=0
                for k in kernel:
                    i_k,j_k=k
                    if image[i+i_k,j+j_k] == 255:
                        contatore+=1
                # se ci sono piu' di 2 pixel nel vicinato non e' rispettata la condizione di continuita',
                # quindi viene inserito nella lista
                if contatore > 2:
                    candidati.append([i,j])
    # TODO: gestire il controllo sulla continuita' dei pixel 
    # parte l'erosione sui pixel candidati
    # precondizione: i pixel candidati non rispettano la continuita'
    while(candidati):
        out=np.zeros((M,N))
        for c in range(0,len(candidati)):
            i_candidati, j_candidati=candidati[c]
            for k in kernel:            
                i_k,j_k=k
                if image[i_candidati+i_k,j_candidati+j_k]==0:
                    trovato=True
                    del candidati[c]
                    c-=1
                    break
                # se tutti i pixel rispettavano i
                if not trovato:
                    out[i,j]=255
        for c in range(0, len(candidati)):
            i_candidati, j_candidati=candidati[c]
            contatore=0
            for k in kernel:            
                i_k,j_k=k
                if out[i_candidati+i_k,j_candidati+j_k] == 255:
                    contatore+=1
                # se ci sono piu' di 2 pixel nel vicinato non e' rispettata la condizione di continuita',
                # quindi viene inserito nella lista
                if contatore < 3:
                    del candidati[c]
                    c-=1
                # TODO: controllo bivio        
        image[:,:]=out[:,:]
    return out


# In[48]:

col=0
image = cv2.imread('coins.jpg',col)
if(col==1):
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# In[52]:

mat=funzioni.hist_cum(image, 256)
binaria=funzioni.otsu_gray(image, mat[0,0,:])
N8=[[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]]
N4=[[0,-1],[-1,0],[1,0],[0,1]]
output=scheletro(binaria)
plt.subplot(121), plt.imshow(binaria,cmap='gray'), plt.title('immagine con poco contrasto')
plt.subplot(122), plt.imshow(output,cmap='gray'), plt.title('immagine con aumento contrasto')


# In[45]:




# In[ ]:



