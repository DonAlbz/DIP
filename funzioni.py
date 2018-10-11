
# coding: utf-8

# In[2]:

#get_ipython().magic(u'matplotlib inline')
import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    

## funzione per flip verticale
def vert_flip(image):
    flipped=np.zeros(image.shape,dtype=np.uint8)
    m=len(image)
    n=len(image[0])
    for i in range(0,m):
        for j in range(0,n):
            flipped[i][j]=image[i][n-1-j]
    return flipped


## funzione per crop
def crop2(image,n1,n2,N1,N2):
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


def convoluzione(image, kernel):
    print "ciao"
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
                    print output[i+k,j+l,:]
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
    #print "ciaoooo"
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


def Lloyd_gray(P, mat, image):
    size=image.shape
    M=len(image)
    N=len(image[0])
    ch=3
    if len(size)==2:
        im_grigia=np.zeros((M,N,1),dtype=np.uint8)
        im_grigia[:,:,0]=image
        image=im_grigia
        ch=1
    
    hist=mat[0,:,:]
    cum=mat[1,:,:]
    
    out=np.zeros((M,N,ch))
    
    for c in range(0,ch):
        print "c vale: ",c
        L=len(hist[c])
        # inizialmente faccio una suddivisione uniforme dell'intervallo (0-L-1) in P intervallini
        delta=L/P
        # considero solo gli estremi "interni", perche' b0 e bP non si spostano, devono essere sempre essere 0 e L
        B = np.zeros(P+1)
        B = B.astype(np.uint8)
        B[0]=0
        B[P]=L-1
        for i in range(1,P):
            B[i]=i*delta
        t=0
        # MSE[1] nuovo, MSE[0] vecchio
        MSE=[-1,-1]
        soglia=0.0001 # 0.01 %
        # inizializzo il rapp tra il vecchio MSE e il nuovo ad un valore "alto"
        rapp=2
        Q=np.zeros(L).astype(np.uint8)
        while np.abs(1-rapp) > 0.0001 and t < 100:
            MSE[0]=MSE[1]
            # calcolo dei centroidi
            V=np.zeros(P) # P = len(B)-1
            for k in range(1,P+1):
                # calcolo prob di normalizzazione
                norm=cum[c,B[k]-1]-cum[c,B[k-1]]+hist[c,B[k-1]]
                # media sull'intervallo
                media=0
                for l in range(B[k-1],B[k]):
                    media+=l*hist[c,l]
                media=media/norm
                V[k-1]=media
            # calcolo la funzione di quantizzazione
            for k in range(0,P):
                Q[B[k]:B[k+1]]=V[k]
            # se no non becca l'ultimo
            Q[-1]=V[P-1]
            # calcolo i nuovi estremi
            for k in range(1,P):
                B[k]=(V[k]+V[k-1])/2
            # calcolo del MSE con formula "furba"
            MSE[1]=0
            for l in range(0,L):
                dif=l-Q[l]
                MSE[1]=MSE[1]+dif*dif*hist[c,l]
            rapp=MSE[1]/MSE[0]
            t=t+1
        for l in range(0,L):
            mask=image[:,:,c]==l
            out[mask,c]=Q[l]
        print V
    
    if(ch==1):
        out_grigia=np.zeros((M,N))
        out_grigia=out[:,:,0]
        out=out_grigia
    return out.astype(np.uint8)


def otsu_gray(image, hist):
    M=len(image)
    N=len(image[0])
    # 0 background, 1 foreground
    out=np.ones((M,N))
    L=len(hist)
    # escludiamo il primo e ultimo valore
    var_min=255*255
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
    return funzioni.crop(S,1,1,M,N)

def split(nodo, image, T):
    # il caso in cui dovrei dividere una regione unitaria e' gia' conteplato nella var maggiore di T
    # se T e' maggiore di 0
    if nodo.dvs > T:
        #tl
        m1=nodo.m/2
        n1=nodo.n/2        
        tlp1=nodo.tlp
        cropped1=image[tlp1[0]:tlp1[0]+m1,tlp1[1]:tlp1[1]+n1]
        mu1=np.mean(cropped1)
        dvs1=np.std(cropped1)
        nodo.tl=NodoAlbero(tlp1,m1,n1,nodo,mu1,dvs1)
        split(nodo.tl, image, T)
        #tr
        m2=m1
        n2=nodo.n-n1        
        tlp2=[nodo.tlp[0], nodo.tlp[1]+n1]
        cropped2=image[tlp2[0]:tlp2[0]+m2,tlp2[1]:tlp2[1]+n2]
        mu2=np.mean(cropped2)
        dvs2=np.std(cropped2)
        nodo.tr=NodoAlbero(tlp2,m2,n2,nodo,mu2,dvs2)
        split(nodo.tr, image, T)
        #bl
        m3=nodo.m-m1
        n3=n1        
        tlp3=[nodo.tlp[0]+m1,nodo.tlp[1]]
        cropped3=image[tlp3[0]:tlp3[0]+m3,tlp3[1]:tlp3[1]+n3]
        mu3=np.mean(cropped3)
        dvs3=np.std(cropped3)
        nodo.bl=NodoAlbero(tlp3,m3,n3,nodo,mu3,dvs3)
        split(nodo.bl, image, T)
        #br
        m4=m3
        n4=n2        
        tlp4=[nodo.tlp[0]+m1,nodo.tlp[1]+n1]
        cropped4=image[tlp4[0]:tlp4[0]+m4,tlp4[1]:tlp4[1]+n4]
        mu4=np.mean(cropped4)
        dvs4=np.std(cropped4)
        nodo.br=NodoAlbero(tlp4,m4,n4,nodo,mu4,dvs4)
        split(nodo.br, image, T)

        
def next_fratello(nodo):
    # caso radice
    if nodo.padre == None:
        return None
    if nodo.padre.tl == nodo:
        return nodo.padre.tr
    if nodo.padre.tr == nodo:
        return nodo.padre.bl
    if nodo.padre.bl == nodo:
        return nodo.padre.br
    # caso in cui nodo è il br
    return None


def next_foglia(foglia):
    x=next_fratello(foglia)
    if x!=None:
        while x.tl!=None:
            x=x.tl
        return x
    x=foglia.padre
    while x!=None and next_fratello(x) == None:
        x=x.padre
        
    if x==None:
        return None
    x=next_fratello(x)
    while x.tl!=None:
        x=x.tl
    return x


def raccogli_foglie(radice):
    A=[]
    x=radice
    while x.tl!=None:
        x=x.tl
    A.append(x) # x punta alla prima foglia
    x=next_foglia(x)
    while x!=None:
        A.append(x)
        x=next_foglia(x)
    return A


# la prima chiamata è con la prima foglia e A=[]
def find_leaves(node,A):
    if node!=node.padre.br:
        if node==node.padre.tl:
            # controllo se ha figli
            if node.padre.tr.tl!=None:
                find_leaves(node.padre.tr.tl,A)
            else:
                A.append(node.padre.tr)
                find_leaves(node.padre.tr,A)
        elif node==node.padre.tr:
            # controllo se ha figli
            if node.padre.bl.tl!=None:
                find_leaves(node.padre.bl.tl,A)
            else:
                A.append(node.padre.bl)
                find_leaves(node.padre.bl,A)
        # controllo se ha figli
        elif node.padre.br.tl!=None:
            find_leaves(node.padre.br.tl,A)
        else:
            A.append(node.padre.br)
            find_leaves(node.padre.br,A)
    else:
        A.append(node)


def find_leaves2(nodo, A):
    if nodo.tl==None:
        A.append(nodo)
    else:
        find_leaves2(nodo.tl,A)
        find_leaves2(nodo.tr,A)
        find_leaves2(nodo.bl,A)
        find_leaves2(nodo.br,A)
        

def merge(foglie, soglia):
    lista=[]
    for i in range(0,len(foglie)):
        lista.append([-1,foglie[i]])
    for i in range(0,len(lista)-1):
        #print "i: ",i
        if lista[i][0]==-1:
            lista[i][0]=1
            for j in range(i+1,len(lista)):
                if lista[j][0] == -1 and abs(lista[j][1].media-lista[i][1].media)<soglia:
                    lista[j][0]=1
                    lista[j][1].media=lista[i][1].media
    

    
    
def disegna_albero(foglie, M, N):
    out=np.zeros((M,N))
    for i in range(0,len(foglie)):
        r=foglie[i].tlp[0]
        c=foglie[i].tlp[1]
        out[r:r+foglie[i].m,c:c+foglie[i].n]=foglie[i].media
    return out
     

class NodoAlbero(object):
    # tlp = top level corner, dvs=deviazione standard
    def __init__ (self, tlp, m, n, padre, media, dvs):
        self.tlp=tlp
        self.m=m
        self.n=n
        self.padre=padre
        self.media=media
        self.dvs=dvs
        self.tl=None
        self.tr=None
        self.bl=None
        self.br=None

        
        
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


# In[ ]:



