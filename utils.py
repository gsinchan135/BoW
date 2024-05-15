from skimage import io
from skimage.util import img_as_float
from skimage.color import rgb2gray
import numpy as np
from scipy.ndimage import correlate
import sklearn.cluster
from scipy.spatial.distance import cdist

def createTextons(F, file_list, K):

    responses = []

    for img_file in file_list:
        img = img_as_float(io.imread(img_file))

        if img.ndim == 3:  #grayscale images
            img = rgb2gray(img)

        for filter in F.T: #apply filter and save response
            responses.append(correlate(img, filter))

    responses = np.array(responses)

    #randomize responses and grab first 100 for samples
    np.random.shuffle(responses)
    samples = responses[:K*100]

    #K-means clustering
    kmeans = sklearn.cluster.KMeans(n_clusters=K)
    kmeans.fit(samples.reshape(-1, responses.shape[-1]))  #compute K-means clustering
    textons = kmeans.cluster_centers_ #coordinates of cluster center

    return textons

def computeHistogram(img_file, F, textons):

    responses = []

    img = img_as_float(io.imread(img_file))
    if img.ndim == 3:   #grayscale images
        img = rgb2gray(img)

    for filter in F.T:  #apply filter and save responses
        responses.append(correlate(img,filter))

    responses = np.array(responses)
    responses = responses.reshape(-1, responses.shape[-1])

    #compare distance from image responses to each texton
    distances = cdist(responses, textons)

    #assigns texton to closest cluster
    closeness = np.argmin(distances, axis=1)

    BoW, _ = np.histogram(closeness, bins=len(textons), range=(0, len(textons)))

    return BoW

