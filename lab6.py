
# coding: utf-8

# In[2]:

get_ipython().magic(u'matplotlib inline')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import funzioni

def canny_edge_detection(image, percTl, percTh, mediato):
    M=len(image)
    N=len(image[0])
    # matrice di output, 0 -> edge, 1 -> non edge; inizializzata a "tutto edge", poi viene via via sbiancata
    out=np.zeros((M,N))
    
    # step 1: eliminazione rumore con filtro gaussiano
    gauss_ker=get_ker_gauss()
    if mediato==1:
        im_med=funzioni.filtraggio(image,gauss_ker,"prop")
        print "filtraggio passa basso finito"
    else:
        im_med=image
    # step 2: calcolare il gradiente dell'immagine
    dx_ker=get_ker_dx()
    dy_ker=get_ker_dy()
    Gx=funzioni.filtraggio_per_gradiente(im_med,dx_ker,"prop")
    print "Gx calcolato"
    Gy=funzioni.filtraggio_per_gradiente(im_med,dy_ker,"prop")
    print "Gy calcolato"
    # modulo del gradiente
    #G=np.sqrt(np.square(Gx)+np.square(Gy))
    G=np.abs(Gx)+np.abs(Gy)
    # direzione del gradiente, in gradi
    theta=np.arctan2(Gy,Gx)*180/np.pi
    # pre-processing con soglia bassa
    massimo=np.max(G)
    print massimo
    Tl=percTl*massimo
    Th=percTh*massimo
    mask=G<=Tl
    out[mask]=255
    # creo una lista con i pixel che verranno analizzati, per risparmiare tempo
    pixel_candidati=[]
    for i in range(2,M-2):
        for j in range(2,N-2):
            if(G[i,j]>Tl and G[i,j]<Th):
                pixel_candidati.append([i,j])
                
    # step 3: considerare i pixel vicini secondo la direzione del gradiente
    k=0
    while k < len(pixel_candidati):
        i,j=pixel_candidati[k]
        if (theta[i,j]>22.5 and theta[i,j]<=67.5) or (theta[i,j]>-157.5 and theta[i,j]<=-112.5):
            # caso "diagonale 45": se il pixel centrale non ha il gradiente maggiore dei 3 pixel vuol dire che non è al "centro" dell'edge
            if G[i,j]<G[i+1,j-1] or G[i,j]<G[i-1,j+1]:
                # marco i,j come non edge -> bianco
                out[i,j]=255
        elif (theta[i,j]>67.5 and theta[i,j]<=112.5) or (theta[i,j]>-112.5 and theta[i,j]<=-67.5):
            # caso "verticale": se il pixel centrale non ha il gradiente maggiore dei 3 pixel vuol dire che non è al "centro" dell'edge
            if G[i,j]<G[i+1,j] or G[i,j]<G[i-1,j]:
                # marco i,j come non edge -> bianco
                out[i,j]=255
        elif (theta[i,j]>112.5 and theta[i,j]<=157.5) or (theta[i,j]>-67.5 and theta[i,j]<=-22.5):
            # caso "diagonale 135": se il pixel centrale non ha il gradiente maggiore dei 3 pixel vuol dire che non è al "centro" dell'edge
            if G[i,j]<G[i+1,j+1] or G[i,j]<G[i-1,j-1]:
                # marco i,j come non edge -> bianco
                out[i,j]=255
        elif (theta[i,j]>-22.5 and theta[i,j] <= 22.5) or (theta[i,j]>157.5 and theta[i,j]<=180) or (theta[i,j]>-180 and theta[i,j]<=-157.5):
            # caso "orizzontale": se il pixel centrale non ha il gradiente maggiore dei 3 pixel vuol dire che non è al "centro" dell'edge
            if G[i,j]<G[i,j-1] or G[i,j]<G[i,j+1]:
                # marco i,j come non edge -> bianco
                out[i,j]=255
        if out[i,j]==255:
            del pixel_candidati[k]
        else:
            k=k+1
            
    # step 4: intorni
    lintorno_piccolo=[[0,0],[0,1],[0,2],[1,0],[1,2],[2,0],[2,1],[2,2]]
    lintorno_grande=[[0,0],[0,1],[0,2],[0,3],[0,4],[1,0],[2,0],[3,0],[4,0],[4,1],[4,2],[4,3],[4,4],[1,4],[2,4],[3,4]]
    k=0
    while k < len(pixel_candidati):
        i,j=pixel_candidati[k]
        intorno=funzioni.crop(G,i-1,j-1,3,3)
        # condizione che verifica che NESSUN pixel nell'intorno abbia gradiente >= Th, inizializzo come
        # "nessun pixel nell'intorno ha gradiente maggiore di Th"
        condizione_no_sopra_th=True
        # condizione che verifica che ALMENO un pixel abbia gradiente maggiore di Tl, inizializzo come
        # "nessun pixel nell'intorno ha gradiente maggiore di Tl"
        condizione_sopra_tl=False
        # 1o controllo
        for l in lintorno_piccolo:
            i_int,j_int=l
            if not condizione_sopra_tl and intorno[i_int,j_int]>Tl:
                condizione_sopra_tl=True
            if intorno[i_int,j_int]>=Th:
                condizione_no_sopra_th=False
                break
        # 2o controllo: lo faccio solo se nessuno dei pixel dell'intorno piccolo aveva gradiente > Th e almeno uno 
        # aveva gradiente compreso tra tl e th, cioè se condizione_sopra_tl=true and condizione_no_sopra_th=true
        if condizione_no_sopra_th and condizione_sopra_tl:
            # condizione che verifica che NESSUN pixel nell'intorno grande abbia gradiente >= Th, inizializzo come
            # "nessun pixel nell'intorno grande ha gradiente maggiore di Th"
            condizione_no_sopra_th_grande=True
            intorno=funzioni.crop(G,i-2,j-2,5,5)
            for l in lintorno_grande:
                i_int,j_int=l
                if intorno[i_int,j_int]>=Th:
                    condizione_no_sopra_th_grande=False
                    break
            if condizione_no_sopra_th_grande:
                # marco i,j come "non edge"
                out[i,j]=255
                del pixel_candidati[k]
                # decremento k, avendo eliminato un elemento dalla lista
                k=k-1
        k=k+1
    return out
            
     
def get_ker_gauss():
    ker=np.zeros((5,5))
    ker[0,0]=2
    ker[0,1]=4
    ker[0,2]=5
    ker[1,0]=4
    ker[1,1]=9
    ker[1,2]=12
    ker[2,0]=5
    ker[2,1]=12
    ker[2,2]=15
    ker[0:3,3:]=ker[0:3,1::-1]
    ker[3:,:]=ker[1::-1,:]
    ker=1/159.0*ker
    return ker


def get_ker_dx():
    dx_ker=np.zeros((3,3))
    dx_ker[:,0]=[-1,-2,-1]
    dx_ker[:,2]=[1,2,1]
    return dx_ker


def get_ker_dy():
    dy_ker=np.zeros((3,3))
    dy_ker[0,:]=[1,2,1]
    dy_ker[2,:]=[-1,-2,-1]
    return dy_ker
    
    


# In[3]:

col=0
image = cv2.imread('picasso.jpg',col)
if(col==1):
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# In[4]:

mediato=1
Tl=0.1
Th=0.2
output=canny_edge_detection(image,Tl,Th,mediato)
plt.imshow(output,cmap='gray')


# In[31]:

dim=100
tgr=255*np.ones((dim,dim))
for i in range(0,dim):
    for j in range(dim-i-1,dim):
        tgr[i,j]=0
cv2.imwrite('tgr.jpg',tgr)
rett=np.zeros((dim,dim))
rett[0:dim/2,:]=255
cv2.imwrite('rett.jpg',rett)


# In[ ]:




# In[5]:

cv2.imwrite('picasso_edge.jpg',output)


# In[7]:

A=np.zeros((100,100))
A[50,40]=200
print np.max(A)


# In[12]:

A=np.zeros((5,5))
A[2,2]=1
A[2,4]=1
mask=A==1
if mask.any():
    print "ciao"


# In[38]:

A=[[1,0],[0,1]]
i,j=A[0]
print i,j


# In[57]:




# In[ ]:



