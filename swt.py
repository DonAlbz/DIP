# -*- encoding: utf-8 -*-
from __future__ import division
from collections import defaultdict
import hashlib
import math
import matplotlib.pyplot as plt
import os
import time
from urllib2 import urlopen

import numpy as np
import cv2
import scipy.sparse, scipy.spatial
import funzioni

t0 = time.clock()

diagnostics = True





class SWTScrubber(object):
    @classmethod
    def scrub(cls, filepath):
        """
        Apply Stroke-Width Transform to image.
        :param filepath: relative or absolute filepath to source image
        :return: numpy array representing result of transform
        """
        canny, sobelx, sobely, theta = cls._create_derivative(filepath)
        '''
        print "dimensioni edges y,x",len(canny[0]), len(canny)
        print "theta matrix:"
        print theta[140:150,0:5]
        '''

        #print canny[143,0]
        gradiente=1 #il gradiente deve essere uguale a 1 (trova lettere chiare) o a -1
        swt , rays2 = cls._swt(theta, canny, sobelx, sobely, gradiente)
        swtSporco = cls._swt2(theta, canny, sobelx, sobely, -gradiente)#salva il risultato dello sporco (che non trova le lettere, ma trova il resto)
        swt[swt==np.Infinity]=0
        canny=cv2.morphologyEx(canny, cv2.MORPH_DILATE, (3,3))
        canny=cv2.morphologyEx(canny, cv2.MORPH_CLOSE, (3,3))
        canny=cv2.morphologyEx(canny, cv2.MORPH_ERODE, (3,3))
        cv2.imwrite('edgesDopoDoppioClose.jpg', canny)
        edgesANDswt=swtSporco+canny
        ret,labels=cv2.connectedComponents(funzioni.negativo(edgesANDswt), connectivity=4)
        print labels.shape
        labelHist=cls.trovaLettere(theta, rays2, labels)
        #viaualizzazione del connected components a colori
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch=255*np.ones_like(label_hue)
        labeled_img=cv2.merge([label_hue, blank_ch, blank_ch])
        labeled_img=cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        labeled_img[label_hue==0]=0
        #print labels[10:15, 10:15]

        labeled_img=cls.coloraAree(labelHist,labels)
        cv2.imwrite('edgesConRegioni.jpg', labeled_img)


       # shapes = cls._connect_components(swt)

        #swts, heights, widths, topleft_pts, images = cls._find_letters(swt, shapes)
        '''
        word_images = cls._find_words(swts, heights, widths, topleft_pts, images)
        '''
        final_mask = np.zeros(swt.shape)
        '''
        kernel=np.ones((3, 3), np.float32)/9
        final_mask=cv2.filter2D(swt,-1, kernel)
        '''
        final_mask=swt
        #final_mask=cv2.morphologyEx(swt, cv2.MORPH_CLOSE, (3,3)) # Applico un closing con un kernel 3x3
        '''
        for word in word_images:
            final_mask += word
        '''
        return final_mask

    @classmethod
    def _create_derivative(cls, filepath):
        img = cv2.imread(filepath,0)
        '''tentativo di incorniciamento: FALLITO
        img=funzioni.incornicia(img)
        print img[len(img)-6:len(img),len(img[0])-10:len(img[0])]
        '''
        edges = cv2.Canny(img, 175, 320, apertureSize=3, L2gradient = False)
        #edges=cv2.imread("fusione_swt.jpg",0)

        # Create gradient map using Sobel
        sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=-1)
        sobely64f = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=-1)
        print 'sobel finiti'

        theta = np.arctan2(sobely64f, sobelx64f)
        if diagnostics:
            cv2.imwrite('edges.jpg',edges)
            cv2.imwrite('sobelx64f.jpg', np.absolute(sobelx64f))
            cv2.imwrite('sobely64f.jpg', np.absolute(sobely64f))
            # amplify theta for visual inspection
            theta_visible = (theta + np.pi)*255/(2*np.pi)
            cv2.imwrite('theta.jpg', theta_visible)
        return (edges, sobelx64f, sobely64f, theta)

    @classmethod
    def coloraAree(cls, labelHist, oldLabel):
        size=oldLabel.shape
        M=len(oldLabel)
        N=len(oldLabel[0])
        if(len(size)==2):
            col=1
            labels=np.zeros((M,N,col))
        for i in range(len(labelHist)):
            if labelHist[i]>0:
                labels[np.where(oldLabel==labelHist[i])]=i+2

        #labels=labels.tolist()
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch=255*np.ones_like(label_hue)
        labeled_img=cv2.merge([label_hue, blank_ch, blank_ch])
        labeled_img=cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        labeled_img[np.where(label_hue==0)]=0
        return labeled_img

    @classmethod
    def trovaLettere(cls, theta, rays2, labels):
        diagonal=(int)(np.sqrt(len(theta)*len(theta) + len(theta[0])*len(theta[0])))
        histRay=np.zeros((diagonal),dtype=np.uint8)
        for i in range(0, len(rays2)):# crea istogramma dei raggi
            histRay[rays2[i][1]] = histRay[rays2[i][1]] + 1


        lettere=[] #array che contiene i raggi con spessore simile
        #boolHist=np.zeros(len(histRay), dtype=bool)

        percentualeIntervallo=0.025
        i=0
        while histRay.any():
            centerHist=np.where(histRay==max(histRay))[0][0] #ritorna l'indice del massimo
            intervallo=(int)(histRay[centerHist]*percentualeIntervallo) #intervallo di tolleranza dal centro agli estremi

            supHist = centerHist + intervallo
            if supHist> len(histRay):
                supHist=len(histRay)
            infHist = centerHist - intervallo
            if infHist<0:
                infHist=0
            while histRay[supHist+1]<histRay[supHist]:
                supHist=supHist+1
            while histRay[infHist-1]<histRay[infHist]:
                infHist=infHist-1
            while histRay[supHist]==0:
                supHist=supHist-1
            while histRay[infHist]==0:
                infHist=infHist+1
            #boolHist[infHist:supHist+1]=True
            lettere.append([np.copy(histRay[infHist:supHist+1]),range(infHist,supHist+1)])
            '''
            struttura lettere = [array1([freq1, freq2, freq3],[spessore1, spessore2,spessore3]), array2([...],[...]),...]
            struttura lettere[0] = [array([freq1,freq2, freq3],[spessore1, spessore2, spessore3])]
            struttura lettere[0][0] = [freq1, freq2, freq3]
            '''
            histRay[infHist:supHist+1]=0
            #print lettere[i]
            i=i+1
        '''
        print "lettere[0]", lettere[0]
        print "lettere[0][0]",lettere[0][0]
        print "lettere[0][1][0]",lettere[0][1][0]
        print "rays2[:]", np.where(np.array(rays2)[:,1]==10) #lista degli indici dei raggi contenuti in ray2 con spessore 10
        #ad ogni elemento di rays2 corrisponde un raggio (ovvero una lista di punti) e lo spessore
        print "rays2[17896]", np.where(np.array(rays2)[:,1]==10)[0]
        '''
        print "len labels, len labels[0]", len(labels), len(labels[0])
        print
        print "len theta, len theta[0]", len(theta), len(theta[0])
        maxLabel=0#massimo numero di label
        for riga in labels:
            if max(riga)>maxLabel:
                maxLabel=max(riga)

        labelHist=np.zeros((maxLabel), dtype=np.uint8)
        print len(labelHist)
        totaleIntorniPresi=(int)( len(lettere)*1) #numero di intorni di lettere presi
        try:
            for i in range (0, totaleIntorniPresi+1): #per ogni array di lettere (intorno) preso
                for spessore in lettere[i][1]: #si prende solamente gli spessori e si tralasciano le frequenze
                    for indiceDiRays2 in np.where(np.array(rays2)[:,1]==spessore)[0]: #per ogni valore dello spessore si risale agli indici dei raggi dell'array rays2 che hanno lo stesso spessore
                       coordinatePuntoMedio=rays2[indiceDiRays2][0][(int)(len(rays2[indiceDiRays2][0])/2)] # si prendono i punti medi di ogni raggio
                       numeroLabel=labels[coordinatePuntoMedio[1],coordinatePuntoMedio[0]] #prendo il numero della label dove si trova il punto medio
                       labelHist[numeroLabel]= labelHist[numeroLabel] + 1 #viene incrementata di 1 la frequenza dei punti presenti sul label considerato
        except IndexError:
            print "eccezione",rays2[indiceDiRays2][0]
        print labelHist
        '''TODO: creare un istogramma dei label ottenuti dal connected components'''
        '''TODO: bug nei raggi trovati, sforano di 1 il bordo dell'immagine.
        '''

        '''
        print max(histRay)
        for i in range(0,len(histRay)):
            if histRay[i]==max(histRay):
                print i
        print np.where(histRay==250)[0][0]
        '''
        return labelHist




    @classmethod
    def _swt(self, theta, edges, sobelx64f, sobely64f, gradiente):
        # create empty image, initialized to infinity
        conta=0
        swt = np.empty(theta.shape)
        swt[:] = np.Infinity
        rays = []
        rays2=[]
        tolleranza=np.pi/4
        diagonal=(int)(np.sqrt(len(theta)*len(theta) + len(theta[0])*len(theta[0])))
        histRay=np.zeros((diagonal),dtype=np.uint8)
        inizio_swt= time.clock() - t0

        # now iterate over pixels in image, checking Canny to see if we're on an edge.
        # if we are, follow a normal a ray to either the next edge or image border
        # edgesSparse = scipy.sparse.coo_matrix(edges)
        step_x_g = gradiente * sobelx64f
        step_y_g = gradiente * sobely64f
        mag_g = np.sqrt(step_x_g * step_x_g + step_y_g * step_y_g) +1

        grad_x_g = step_x_g / mag_g
        grad_y_g = step_y_g / mag_g


        for x in xrange(edges.shape[1]):
            for y in xrange(edges.shape[0]):
                if edges[y, x] > 0:
                    step_x = step_x_g[y, x]
                    step_y = step_y_g[y, x]
                    mag = mag_g[y, x]
                    grad_x = grad_x_g[y, x]
                    grad_y = grad_y_g[y, x]
                    ray = [] #raggio
                    ray.append((x, y))

                    prev_x, prev_y, i = x, y, 0
                    noBordi= True
                    while noBordi:

                        i += 1
                        cur_x = int(math.floor(x + grad_x * i))
                        cur_y = int(math.floor(y + grad_y * i))
                        '''
                        print " "
                        print "i:"
                        print i
                        print "x:"
                        print x
                        print "grad_x:"
                        print grad_x
                        print "cur_x:"
                        print cur_x
                        print " "
                        print "y:"
                        print y
                        print "grad_y:"
                        print grad_y
                        print "cur_y: "
                        print cur_y
                        print "edges[cur_y, cur_x]"
                       #print edges[cur_y, cur_x]
                        print "-------------------------------------------- "
                        '''
                        if cur_x != prev_x or cur_y != prev_y:
                            # we have moved to the next pixel!
                            '''
                            print " "
                            print "edges:"
                            
                            print " "
                    
                            print "-------------------------------------------- "
                            '''
                            try:
                                conta=conta+1
                                if cur_y>0 and cur_x>0:
                                    if edges[cur_y, cur_x] > 0:
                                    #if edges[cur_y, cur_x] > 0:
                                        # found edge in the moved position
                                        #QUI NON ENTRA
                                        ray.append((cur_x, cur_y))
                                        theta_point = theta[y, x]
                                        alpha = theta[cur_y, cur_x]
                                        #if math.acos(grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x]) < np.pi/2.0:
                                        if abs(abs(alpha-theta_point)-np.pi)<tolleranza:
                                            thickness = math.sqrt( (cur_x - x)*(cur_x-x) + (cur_y - y)*(cur_y-y) )
                                            for (rp_x, rp_y) in ray:
                                                swt[rp_y, rp_x] = min(thickness, swt[rp_y, rp_x])
                                            rays.append(ray)
                                            rays2.append([ray,(int)(thickness)])
                                        break
                                else:
                                    noBordi=False
                                # this is positioned at end to ensure we don't add a point beyond image boundary
                                ray.append((cur_x, cur_y))
                            except IndexError:

                                break
                            prev_x = cur_x
                            prev_y = cur_y

        # Compute median SWT
        for ray in rays:
            median = np.median([swt[y, x] for (x, y) in ray])
            for (x, y) in ray:
                swt[y, x] = min(median, swt[y, x])
        if diagnostics:
            cv2.imwrite('swt.jpg', swt)
            print "tempo totale swt in secondi:"
            fine_swt = time.clock() - t0
            print fine_swt-inizio_swt
            print "conta=", conta
        '''
        for i in range(0, len(rays2)):# crea istogramma dei raggi
            histRay[rays2[i][1]] = histRay[rays2[i][1]] + 1


        lettere=[] #array che contiene i raggi con spessore simile
        #boolHist=np.zeros(len(histRay), dtype=bool)

        percentualeIntervallo=0.025
        i=0
        while histRay.any():
            centerHist=np.where(histRay==max(histRay))[0][0] #ritorna l'indice del massimo
            intervallo=(int)(histRay[centerHist]*percentualeIntervallo) #intervallo di tolleranza dal centro agli estremi

            supHist = centerHist + intervallo
            if supHist> len(histRay):
                supHist=len(histRay)
            infHist = centerHist - intervallo
            if infHist<0:
                infHist=0
            while histRay[supHist+1]<histRay[supHist]:
                supHist=supHist+1
            while histRay[infHist-1]<histRay[infHist]:
                infHist=infHist-1
            while histRay[supHist]==0:
                supHist=supHist-1
            while histRay[infHist]==0:
                infHist=infHist+1
            #boolHist[infHist:supHist+1]=True
            lettere.append([np.copy(histRay[infHist:supHist+1]),range(infHist,supHist+1)])
        '''
        '''
            struttura lettere = [array1([freq1, freq2, freq3],[spessore1, spessore2,spessore3]), array2([...],[...]),...]
            struttura lettere[0] = [array([freq1,freq2, freq3],[spessore1, spessore2, spessore3])]
            struttura lettere[0][0] = [freq1, freq2, freq3]
        '''
        '''
            histRay[infHist:supHist+1]=0
            #print lettere[i]
            i=i+1
        '''
        '''
        print "lettere[0]", lettere[0]
        print "lettere[0][0]",lettere[0][0]
        print "lettere[0][1][0]",lettere[0][1][0]
        print "rays2[:]", np.where(np.array(rays2)[:,1]==10) #lista degli indici dei raggi contenuti in ray2 con spessore 10
        #ad ogni indice corrisponde un raggio (ovvero una lista di punti) e lo spessore
        '''
        '''
        print "rays2[17896]", np.where(np.array(rays2)[:,1]==10)[0]

        totaleIntorniPresi=(int)( len(lettere)*0.1) #numero di intorni di lettere presi
        for i in range (0, totaleIntorniPresi+1): #per ogni array di lettere (intorno) preso
            for spessore in lettere[i][1]: #si prende solamente gli spessori e si tralasciano le frequenze
                #print "spessore", spessore
                for indiceDiRays2 in np.where(np.array(rays2)[:,1]==spessore)[0]: #per ogni valore dello spessore si risale agli indici dei raggi dell'array rays2 che hanno lo stesso spessore
                   #print rays2[indiceDiRays2][0]
                   print rays2[indiceDiRays2][0][(int)(len(rays2[indiceDiRays2][0])/2)] # si prendono i punti medi di ogni raggio


        '''
        '''
        print max(histRay)
        for i in range(0,len(histRay)):
            if histRay[i]==max(histRay):
                print i
        print np.where(histRay==250)[0][0]
        '''

        return swt, rays2

    #swt sporca che trova quello che non è una lettera
    @classmethod
    def _swt2(self, theta, edges, sobelx64f, sobely64f, gradiente):
        # create empty image, initialized to infinity
        swt = np.empty(theta.shape)
        swt[:] = np.Infinity
        rays = []
        tolleranza=np.pi/2

        inizio_swt= time.clock() - t0

        # now iterate over pixels in image, checking Canny to see if we're on an edge.
        # if we are, follow a normal a ray to either the next edge or image border
        # edgesSparse = scipy.sparse.coo_matrix(edges)
        step_x_g = gradiente * sobelx64f
        step_y_g = gradiente * sobely64f
        mag_g = np.sqrt(step_x_g * step_x_g + step_y_g * step_y_g) +1

        grad_x_g = step_x_g / mag_g
        grad_y_g = step_y_g / mag_g


        for x in xrange(edges.shape[1]):
            for y in xrange(edges.shape[0]):
                if edges[y, x] > 0:
                    step_x = step_x_g[y, x]
                    step_y = step_y_g[y, x]
                    mag = mag_g[y, x]
                    grad_x = grad_x_g[y, x]
                    grad_y = grad_y_g[y, x]
                    ray = [] #raggio
                    ray.append((x, y))
                    prev_x, prev_y, i = x, y, 0
                    while True:

                        i += 1
                        cur_x = int(math.floor(x + grad_x * i))
                        cur_y = int(math.floor(y + grad_y * i))
                        '''
                        print " "
                        print "i:"
                        print i
                        print "x:"
                        print x
                        print "grad_x:"
                        print grad_x
                        print "cur_x:"
                        print cur_x
                        print " "
                        print "y:"
                        print y
                        print "grad_y:"
                        print grad_y
                        print "cur_y: "
                        print cur_y
                        print "edges[cur_y, cur_x]"
                       #print edges[cur_y, cur_x]
                        print "-------------------------------------------- "
                        '''
                        if cur_x != prev_x or cur_y != prev_y:
                            # we have moved to the next pixel!
                            '''
                            print " "
                            print "edges:"
                            
                            print " "
                    
                            print "-------------------------------------------- "
                            '''
                            try:
                                if edges[cur_y, cur_x] > 0:
                                    # found edge in the moved position
                                    #QUI NON ENTRA
                                    ray.append((cur_x, cur_y))
                                    theta_point = theta[y, x]
                                    alpha = theta[cur_y, cur_x]
                                    #if math.acos(grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x]) < np.pi/2.0:
                                    if abs(abs(alpha-theta_point)-np.pi)<tolleranza:
                                        #thickness = math.sqrt( (cur_x - x)*(cur_x-x) + (cur_y - y)*(cur_y-y) )
                                        for (rp_x, rp_y) in ray:
                                            swt[rp_y, rp_x] = 255
                                        rays.append(ray)

                                    break
                                # this is positioned at end to ensure we don't add a point beyond image boundary
                                ray.append((cur_x, cur_y))
                            except IndexError:
                                # reached image boundary

                                theta_point = theta[y, x]
                                alpha = theta[prev_y, prev_x]
                                #if math.acos(grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x]) < np.pi/2.0:
                                if abs(abs(alpha-theta_point)-np.pi)<np.pi*2:
                                    #thickness = math.sqrt( (cur_x - x)*(cur_x-x) + (cur_y - y)*(cur_y-y) )
                                    for (rp_x, rp_y) in ray:
                                        swt[rp_y, rp_x] = 255
                                    rays.append(ray)

                                break
                            prev_x = cur_x
                            prev_y = cur_y
        if diagnostics:
            cv2.imwrite('swtSporco.jpg', swt)
            print "tempo totale swt in secondi:"
            fine_swt = time.clock() - t0
            print fine_swt-inizio_swt
        #trasforma swt da una matrice float ad una matrice di interi
        swt[swt==np.Infinity]=0

        return swt



    @classmethod
    def _connect_components(cls, swt):
        # STEP: Compute distinct connected components
        # Implementation of disjoint-set
        class Label(object):
            def __init__(self, value):
                self.value = value
                self.parent = self
                self.rank = 0
            def __eq__(self, other):
                if type(other) is type(self):
                    return self.value == other.value
                else:
                    return False
            def __ne__(self, other):
                return not self.__eq__(other)

        ld = {}

        def MakeSet(x):
            try:
                return ld[x]
            except KeyError:
                item = Label(x)
                ld[x] = item
                return item

        def Find(item):
            # item = ld[x]
            if item.parent != item:
                item.parent = Find(item.parent)
            return item.parent

        def Union(x, y):
            """
            :param x:
            :param y:
            :return: root node of new union tree
            """
            x_root = Find(x)
            y_root = Find(y)
            if x_root == y_root:
                return x_root

            if x_root.rank < y_root.rank:
                x_root.parent = y_root
                return y_root
            elif x_root.rank > y_root.rank:
                y_root.parent = x_root
                return x_root
            else:
                y_root.parent = x_root
                x_root.rank += 1
                return x_root

        # apply Connected Component algorithm, comparing SWT values.
        # components with a SWT ratio less extreme than 1:3 are assumed to be
        # connected. Apply twice, once for each ray direction/orientation, to
        # allow for dark-on-light and light-on-dark texts
        trees = {}
        # Assumption: we'll never have more than 65535-1 unique components
        label_map = np.zeros(shape=swt.shape, dtype=np.uint16) #matrice che contiene i pixel al valore del label
        next_label = 1
        # First Pass, raster scan-style
        swt_ratio_threshold = 3.0
        for y in xrange(swt.shape[0]):
            for x in xrange(swt.shape[1]):
                sw_point = swt[y, x]
                if sw_point < np.Infinity and sw_point > 0:
                    neighbors = [(y, x-1),   # west
                                 (y-1, x-1), # northwest
                                 (y-1, x),   # north
                                 (y-1, x+1)] # northeast
                    connected_neighbors = None
                    neighborvals = [] # contenuto il valore dei pixel vicini in label_map

                    for neighbor in neighbors:
                        # west
                        try:
                            sw_n = swt[neighbor] #valore del pixel vicino in swt
                            label_n = label_map[neighbor] #valore del pixel vicino in label_map
                        except IndexError:
                            continue
                        if label_n > 0 and sw_n / sw_point < swt_ratio_threshold and sw_point / sw_n < swt_ratio_threshold:
                            neighborvals.append(label_n) #si aggiunge il valore del vicino in label_map
                            if connected_neighbors:
                                connected_neighbors = Union(connected_neighbors, MakeSet(label_n))
                            else:
                                connected_neighbors = MakeSet(label_n)

                    if not connected_neighbors:
                        # We don't see any connections to North/West
                        trees[next_label] = (MakeSet(next_label))
                        label_map[y, x] = next_label
                        next_label += 1
                    else:
                        # We have at least one connection to North/West
                        label_map[y, x] = min(neighborvals)
                        # For each neighbor, make note that their respective connected_neighbors are connected
                        # for label in connected_neighbors. @todo: do I need to loop at all neighbor trees?
                        trees[connected_neighbors.value] = Union(trees[connected_neighbors.value], connected_neighbors)

        # Second pass. re-base all labeling with representative label for each connected tree
        layers = {}
        contours = defaultdict(list)
        for x in xrange(swt.shape[1]):
            for y in xrange(swt.shape[0]):
                if label_map[y, x] > 0:
                    item = ld[label_map[y, x]]
                    common_label = Find(item).value
                    label_map[y, x] = common_label
                    contours[common_label].append([x, y])
                    try:
                        layer = layers[common_label]
                    except KeyError:
                        layers[common_label] = np.zeros(shape=swt.shape, dtype=np.uint16)
                        layer = layers[common_label]

                    layer[y, x] = 1
        return layers

    @classmethod
    def _find_letters(cls, swt, shapes):
        # STEP: Discard shapes that are probably not letters
        swts = []
        heights = []
        widths = []
        topleft_pts = []
        images = []

        for label,layer in shapes.iteritems():
            (nz_y, nz_x) = np.nonzero(layer)
            east, west, south, north = max(nz_x), min(nz_x), max(nz_y), min(nz_y)
            width, height = east - west, south - north

            if width < 8 or height < 8:
                continue

            if width / height > 10 or height / width > 10:
                continue

            diameter = math.sqrt(width * width + height * height)
            median_swt = np.median(swt[(nz_y, nz_x)])
            if diameter / median_swt > 10:
                continue

            if width / layer.shape[1] > 0.4 or height / layer.shape[0] > 0.4:
                continue

            if diagnostics:
                print " written to image."
                cv2.imwrite('layer'+ str(label) +'.jpg', layer * 255)

            # we use log_base_2 so we can do linear distance comparison later using k-d tree
            # ie, if log2(x) - log2(y) > 1, we know that x > 2*y
            # Assumption: we've eliminated anything with median_swt == 1
            swts.append([math.log(median_swt, 2)])
            heights.append([math.log(height, 2)])
            topleft_pts.append(np.asarray([north, west]))
            widths.append(width)
            images.append(layer)

        return swts, heights, widths, topleft_pts, images

    @classmethod
    def _find_words(cls, swts, heights, widths, topleft_pts, images):
        # Find all shape pairs that have similar median stroke widths
        print 'SWTS'
        print swts
        print 'DONESWTS'
        swt_tree = scipy.spatial.KDTree(np.asarray(swts))
        stp = swt_tree.query_pairs(1)

        # Find all shape pairs that have similar heights
        height_tree = scipy.spatial.KDTree(np.asarray(heights))
        htp = height_tree.query_pairs(1)

        # Intersection of valid pairings
        isect = htp.intersection(stp)

        chains = []
        pairs = []
        pair_angles = []
        for pair in isect:
            left = pair[0]
            right = pair[1]
            widest = max(widths[left], widths[right])
            distance = np.linalg.norm(topleft_pts[left] - topleft_pts[right])
            if distance < widest * 3:
                delta_yx = topleft_pts[left] - topleft_pts[right]
                angle = np.arctan2(delta_yx[0], delta_yx[1])
                if angle < 0:
                    angle += np.pi

                pairs.append(pair)
                pair_angles.append(np.asarray([angle]))

        angle_tree = scipy.spatial.KDTree(np.asarray(pair_angles))
        atp = angle_tree.query_pairs(np.pi/12)

        for pair_idx in atp:
            pair_a = pairs[pair_idx[0]]
            pair_b = pairs[pair_idx[1]]
            left_a = pair_a[0]
            right_a = pair_a[1]
            left_b = pair_b[0]
            right_b = pair_b[1]

            # @todo - this is O(n^2) or similar, extremely naive. Use a search tree.
            added = False
            for chain in chains:
                if left_a in chain:
                    chain.add(right_a)
                    added = True
                elif right_a in chain:
                    chain.add(left_a)
                    added = True
            if not added:
                chains.append(set([left_a, right_a]))
            added = False
            for chain in chains:
                if left_b in chain:
                    chain.add(right_b)
                    added = True
                elif right_b in chain:
                    chain.add(left_b)
                    added = True
            if not added:
                chains.append(set([left_b, right_b]))

        word_images = []
        for chain in [c for c in chains if len(c) > 3]:
            for idx in chain:
                word_images.append(images[idx])
                # cv2.imwrite('keeper'+ str(idx) +'.jpg', images[idx] * 255)
                # final += images[idx]

        return word_images



final_mask = SWTScrubber.scrub('img6.jpg')

'''
file_url = 'http://upload.wikimedia.org/wikipedia/commons/0/0b/ReceiptSwiss.jpg'
local_filename = hashlib.sha224(file_url).hexdigest()

try:
    s3_response=1
    
    s3_response = urlopen(file_url)
    with open(local_filename, 'wb+') as destination:
        while True:
            # read file in 4kB chunks
            chunk = s3_response.read(4096)
            if not chunk: break
            destination.write(chunk)
            
    



    #final_mask = SWTScrubber.scrub(local_filename)

    # final_mask = cv2.GaussianBlur(final_mask, (1, 3), 0)
    # cv2.GaussianBlur(sobelx64f, (3, 3), 0)
    cv2.imwrite('final.jpg', final_mask*20)
    print time.clock() - t0

finally:
s3_response.close()
'''
cv2.imwrite('final.jpg', final_mask)

