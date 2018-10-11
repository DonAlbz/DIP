
# coding: utf-8

# In[106]:

get_ipython().magic(u'matplotlib inline')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import funzioni

def scala_ruota_trasla(image, Sx,Sy, gradi, dx,dy, col_c,rig_c):
    # esegue la seguente trasformazione:[x',y']=T*R*S*[x,y], cioè esegue prima una scalatura
    # di Sx,Sy, poi una rotazione di phi e poi una traslazione di dx,dy.
    # la trasformazione inversa (necessaria all'algoritmo) deve dunque essere:
    # traslazione di -dx,-dy, rotazione di -phi e scalatura di 1/Sx, 1/Sy
    # NB: eseguire le 3 traformazioni in un colpo solo è ovviamente più efficiente che fare le 3 trasformazioni
    # separatamente, perchè costruiamo subito la matrice
    size=image.shape
    M=len(image)
    N=len(image[0])
    col=3
    # caso grigio (solito trucco)
    if(len(size)==2):
        col=1
        im_grigia=np.zeros((M,N,col))
        im_grigia[:,:,0]=image
        image=im_grigia
    # dimensione output è maggiore
    out=np.zeros((M,N,col))
    invS=np.eye(3)
    invS[0,0]=1/float(Sx)
    invS[1,1]=1/float(Sy)
    phi=(2*np.pi*gradi)/360.0
    invR=np.zeros((3,3))
    invR[0,0]=np.cos(phi)
    invR[0,1]=np.sin(phi)
    invR[1,0]=-np.sin(phi)
    invR[1,1]=np.cos(phi)
    invR[2,2]=1
    invTras=np.eye(3)
    invTras[0,2]=-dx
    invTras[1,2]=-dy
    invT=np.dot(invS,np.dot(invR,invTras))
    coord_out=np.zeros(3)
    # faccio scorrere tutte le posizioni dei pixel dell'uscita
    for i in range(0,M):
        for j in range(0,N):
            coord_out[0]=j-col_c
            coord_out[1]=rig_c-i
            coord_out[2]=1
            coord_partenza=np.dot(invT,coord_out)
            i_imm_partenza_float=rig_c-coord_partenza[1]
            j_imm_partenza_float=col_c+coord_partenza[0]
            # siccome ho ottenuto delle coordinate in float, ora devo fare una interpolazione dei pixel vicini
            # nell'immagine di partenza
            i_imm_partenza=np.round(i_imm_partenza_float)
            j_imm_partenza=np.round(j_imm_partenza_float)
            sforato=False
            if i_imm_partenza > M-1:
                i_imm_partenza=M-1
                sforato=True
            elif i_imm_partenza <0:
                i_imm_partenza=0
                sforato=True
            if j_imm_partenza > N-1:
                j_imm_partenza=N-1
                sforato=True
            elif j_imm_partenza<0:
                j_imm_partenza=0
                sforato=True
            if not sforato:
                #out[i,j]=image[i_imm_partenza,j_imm_partenza]
                out[i,j,:]=interp_bilinear(image,i_imm_partenza_float,j_imm_partenza_float,len(size)==2)
            else:
                out[i,j,:]=0
    # se l'input era grigio restituisco un'immagine di tipo grayscale
    if(len(size)==2):
        out_grig=np.zeros((M,N))
        out_grig=out[:,:,0]
        out=out_grig
    return out.astype(np.uint8)
    
    
    
    
    
def scaling(image, Sx, Sy, col_c, rig_c):
    size=image.shape
    M=len(image)
    N=len(image[0])
    col=3
    # caso grigio (solito trucco)
    if(len(size)==2):
        col=1
        im_grigia=np.zeros((M,N,col))
        im_grigia[:,:,0]=image
        image=im_grigia
    # dimensione output è maggiore
    out=np.zeros((M,N,col))
    invT=np.eye(2)
    invT[0,0]=1/float(Sx)
    invT[1,1]=1/float(Sy)
    coord_out=np.zeros(2)
    # faccio scorrere tutte le posizioni dei pixel dell'uscita
    for i in range(0,M):
        for j in range(0,N):
            coord_out[0]=j-col_c
            coord_out[1]=rig_c-i
            coord_partenza=np.dot(invT,coord_out)
            i_imm_partenza_float=rig_c-coord_partenza[1]
            j_imm_partenza_float=col_c+coord_partenza[0]
            # siccome ho ottenuto delle coordinate in float, ora devo fare una interpolazione dei pixel vicini
            # nell'immagine di partenza
            i_imm_partenza=np.round(i_imm_partenza_float)
            j_imm_partenza=np.round(j_imm_partenza_float)
            sforato=False
            if i_imm_partenza > M-1:
                i_imm_partenza=M-1
                sforato=True
            elif i_imm_partenza <0:
                i_imm_partenza=0
                sforato=True
            if j_imm_partenza > N-1:
                j_imm_partenza=N-1
                sforato=True
            elif j_imm_partenza<0:
                j_imm_partenza=0
                sforato=True
            if not sforato:
                #out[i,j]=image[i_imm_partenza,j_imm_partenza]
                out[i,j,:]=interp_bilinear(image,i_imm_partenza_float,j_imm_partenza_float,len(size)==2)
            else:
                out[i,j,:]=0
    # se l'input era grigio restituisco un'immagine di tipo grayscale
    if(len(size)==2):
        out_grig=np.zeros((M,N))
        out_grig=out[:,:,0]
        out=out_grig
    return out.astype(np.uint8)


def traslazione(image, dx,dy, col_c, rig_c):
    size=image.shape
    M=len(image)
    N=len(image[0])
    col=3
    # caso grigio (solito trucco)
    if(len(size)==2):
        col=1
        im_grigia=np.zeros((M,N,col))
        im_grigia[:,:,0]=image
        image=im_grigia
    # dimensione output è maggiore
    out=np.zeros((M,N,col))
    invT=np.eye(3)
    invT[0,2]=-dx
    invT[1,2]=-dy
    coord_out=np.zeros(3)
    # faccio scorrere tutte le posizioni dei pixel dell'uscita
    for i in range(0,M):
        for j in range(0,N):
            coord_out[0]=j-col_c
            coord_out[1]=rig_c-i
            coord_out[2]=1
            coord_partenza=np.dot(invT,coord_out)
            i_imm_partenza_float=rig_c-coord_partenza[1]
            j_imm_partenza_float=col_c+coord_partenza[0]
            # siccome ho ottenuto delle coordinate in float, ora devo fare una interpolazione dei pixel vicini
            # nell'immagine di partenza
            i_imm_partenza=np.round(i_imm_partenza_float)
            j_imm_partenza=np.round(j_imm_partenza_float)
            sforato=False
            if i_imm_partenza > M-1:
                i_imm_partenza=M-1
                sforato=True
            elif i_imm_partenza <0:
                i_imm_partenza=0
                sforato=True
            if j_imm_partenza > N-1:
                j_imm_partenza=N-1
                sforato=True
            elif j_imm_partenza<0:
                j_imm_partenza=0
                sforato=True
            if not sforato:
                #out[i,j]=image[i_imm_partenza,j_imm_partenza]
                out[i,j,:]=interp_bilinear(image,i_imm_partenza_float,j_imm_partenza_float,len(size)==2)
            else:
                out[i,j,:]=0
    # se l'input era grigio restituisco un'immagine di tipo grayscale
    if(len(size)==2):
        out_grig=np.zeros((M,N))
        out_grig=out[:,:,0]
        out=out_grig
    return out.astype(np.uint8)
    
    
    
    
def rotazione(image, gradi, col_c, rig_c):
    size=image.shape
    phi=(2*np.pi*gradi)/360.0
    invT=np.zeros((3,3))
    invT[0,0]=np.cos(phi)
    invT[0,1]=np.sin(phi)
    invT[1,0]=-np.sin(phi)
    invT[1,1]=np.cos(phi)
    invT[2,2]=1
    M=len(image)
    N=len(image[0])
    
    col=3
    # caso grigio (solito trucco)
    if(len(size)==2):
        col=1
        im_grigia=np.zeros((M,N,col))
        im_grigia[:,:,0]=image
        image=im_grigia
    # dimensione output è maggiore
    out=np.zeros((M,N,col))
    
    coord_out=np.zeros(3)
    # faccio scorrere tutte le posizioni dei pixel dell'uscita
    for i in range(0,M):
        for j in range(0,N):
            coord_out[0]=j-col_c
            coord_out[1]=rig_c-i
            coord_out[2]=1
            coord_partenza=np.dot(invT,coord_out)
            i_imm_partenza_float=rig_c-coord_partenza[1]
            j_imm_partenza_float=col_c+coord_partenza[0]
            # siccome ho ottenuto delle coordinate in float, ora devo fare una interpolazione dei pixel vicini
            # nell'immagine di partenza
            i_imm_partenza=np.round(i_imm_partenza_float)
            j_imm_partenza=np.round(j_imm_partenza_float)
            sforato=False
            if i_imm_partenza > M-1:
                i_imm_partenza=M-1
                sforato=True
            elif i_imm_partenza <0:
                i_imm_partenza=0
                sforato=True
            if j_imm_partenza > N-1:
                j_imm_partenza=N-1
                sforato=True
            elif j_imm_partenza<0:
                j_imm_partenza=0
                sforato=True
            if not sforato:
                #out[i,j]=image[i_imm_partenza,j_imm_partenza]
                out[i,j,:]=interp_bilinear(image,i_imm_partenza_float,j_imm_partenza_float,len(size)==2)
            else:
                out[i,j,:]=0
    # se l'input era grigio restituisco un'immagine di tipo grayscale
    if(len(size)==2):
        out_grig=np.zeros((M,N))
        out_grig=out[:,:,0]
        out=out_grig
    return out.astype(np.uint8)
    
    
def interp_bilinear(image,i_float,j_float,grigia):
    M=len(image)
    N=len(image[0])
    a=j_float-np.floor(j_float)
    b=i_float-np.floor(i_float)
    col=3
    if grigia:
        col=1
    R1=np.zeros(col)
    R2=np.zeros(col)
    R=np.zeros(col)
    if np.ceil(j_float)<N and np.floor(j_float)>=0 and np.ceil(i_float)<M and np.floor(i_float)>=0:
        R1=a*image[int(np.floor(i_float)), int(np.ceil(j_float)),:]+(1-a)*image[int(np.floor(i_float)), int(np.floor(j_float)),:]
        R2=a*image[int(np.ceil(i_float)), int(np.ceil(j_float)),:]+(1-a)*image[int(np.ceil(i_float)), int(np.floor(j_float)),:]
        R=b*R2+(1-b)*R1
    return R







# In[107]:

col=1
# l'immagine caricata è gia binaria
image = cv2.imread('mont.jpg',col)
if(col==1):
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# In[113]:

M=len(image)
N=len(image[0])
col_c=0
rig_c=0
if M%2==0:
    rig_c=M/2
else:
    rig_c=(M-1)/2
if N%2==0:
    col_c=N/2
else:
    col_c=(N-1)/2
Sx=1.6
Sy=1.3
#trasformata=scaling(image,Sx,Sy,col_c,rig_c)
phi=-130
#trasformata=rotazione(trasformata,phi,col_c,rig_c)
dx=0
dy=0
#trasformata=traslazione(trasformata,dx,dy,400,400)
trasformata=scala_ruota_trasla(image,Sx,Sy, phi, dx,dy, col_c,rig_c)
plt.imshow(trasformata,cmap='gray')


# In[114]:

if col==1:
    trasformata=cv2.cvtColor(trasformata, cv2.COLOR_RGB2BGR)
cv2.imwrite('trasformata.jpg',trasformata)


# In[55]:




# In[ ]:



