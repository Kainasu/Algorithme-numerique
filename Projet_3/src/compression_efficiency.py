import numpy as np
import matplotlib.pyplot as plt
import compression as cmp

# Renvoie la trace de la matrice M
def tr(M):
    res = 0
    for k in range(np.shape(M)[0]):
        res += M[k][k]
    return res

# Renvoie une distance entre img_1 et img_2
def dist_img(img_1, img_2):
    dist = 0
    for k in range(3):
        M = img_1[:,:,k] - img_2[:,:,k]
        dist += tr(np.dot(M.T, M))
    return np.sqrt(dist)
    
  
def distance_img_to_compressed_img(img, k, NMax):
    img_cmp = cmp.compression_rank(k, img, NMax)
    return dist_img(img, img_cmp)

def list_distance(img, NMax):    
    dist = []
    m, n, p = np.shape(img)
    for k in range(min(m, n)):
        dist.append(distance_img_to_compressed_img(img, k, NMax))
    return dist

def graph_distance(img, NMax):
    y = list_distance(img, NMax)
    x = range(len(y))
    plt.xlabel("Rang de compression")
    plt.ylabel("Distance")
    plt.plot(x,y) 
    plt.show()