import numpy as np
import cv2
import torch
import copy
from sklearn.neighbors import KNeighborsClassifier

def obtainDetEmbeddings(model, transforms, trackPath, videoPath, embeddingSize, device = "cuda"):
    """
    This function computes the embeddings of each detection in tracks from the video.

    Parameters
    ----------
    model : pytorch model
        Model to compute the embeddings.
    transforms : pytorch transforms
        Transforms used for model input.
    trackPath : str
        File where the multi-object single-camera tracking results are.
    videoPath : str
        Video path.
    embeddingSize : int
        Size of the embedding that the model outputs per image.
    device : str, optional
        Device to inference the model. The default is "cuda".

    Returns
    -------
    tracksEmbeddings : dict
        Dictionary with 2 keys ["ids", "embeddings"]. Ids refers to the embedding track id.

    """
    
    # Load video
    video = cv2.VideoCapture(videoPath)
    
    # Load multi-object track
    videoTracks = np.loadtxt(trackPath, delimiter=',')
    
    # Init results
    tracksEmbeddings = {}
    tracksEmbeddings["ids"] = []
    tracksEmbeddings["embeddings"] = np.zeros((0,embeddingSize))
    
    
    indexTrackLines = 0
    
    while indexTrackLines < len(videoTracks):
        
        # Get det info
        detIndex = int(videoTracks[indexTrackLines][0])
        detId = int(videoTracks[indexTrackLines][1])
        x = int(float(videoTracks[indexTrackLines][2]))
        y = int(float(videoTracks[indexTrackLines][3]))
        w = int(float(videoTracks[indexTrackLines][4]))
        h = int(float(videoTracks[indexTrackLines][5]))
        
        # Find frame
        frameIndex = video.get(cv2.CV_CAP_PROP_POS_FRAMES)
        
        while  frameIndex != detIndex:
            
            # Get next frame
            _, currentFrame = video.read()
        
            frameIndex = video.get(cv2.CV_CAP_PROP_POS_FRAMES)
        
        # Get crop
        imageCrop = currentFrame[y:y+h, x:x+w]
        imageCrop = cv2.cvtColor(imageCrop, cv2.COLOR_BGR2RGB)
        transformedImageCrop = transforms(imageCrop)
        transformedImageCrop = transformedImageCrop.to(device)
        
        with torch.no_grad():
            # Get embedding
            embedding = model(transformedImageCrop)
        embedding = embedding.cpu().numpy()
        
        # Store
        tracksEmbeddings["embeddings"] = np.vstack((tracksEmbeddings["embeddings"], embedding))
        tracksEmbeddings["ids"].append(detId)
        
        indexTrackLines += 1

    return tracksEmbeddings

def ReIdByCentroids(tracksEmbeddings1, tracksEmbeddings2, disThreshold):
    """
    This function giving the track information of 2 multi-object single-camera trackings results
    re-assigns ids matching different camera tracks. For the matching the centroids comparison is made.

    Parameters
    ----------
    tracksEmbeddings1 : dict
        Reference tracking results.
    tracksEmbeddings2 : dict
        Query tracking results.
    disThreshold : float
        Embedding distance threshold to be considered same track.

    Returns
    -------
    matches : dict
        Assignmests of old query labels and new ones.
    resultTracksEmbeddings : dict
        Updated results with 2 camera results.

    """
    refLabels = np.array(tracksEmbeddings1["ids"]) 
    refUniqueLabels = np.unique(refLabels)
    refEmbeddings = tracksEmbeddings1["embeddings"]
    
    currentId = np.sort(refUniqueLabels)[-1] + 1
    
    # Compute centroids
    refCentroids = np.zeros((0, refEmbeddings.shape[-1]))
    for label in refUniqueLabels:
        centroid = np.mean(refEmbeddings[refLabels == label, :], axis = 0)
        refCentroids = np.vstack((refCentroids, centroid))
    
    queryLabels = np.array(tracksEmbeddings2["ids"]) 
    queryUniqueLabels = np.unique(queryLabels)
    queryEmbeddings = tracksEmbeddings2["embeddings"]
    
    # Compute centroids
    queryCentroids = np.zeros((0, queryEmbeddings.shape[-1]))
    for label in queryUniqueLabels:
        centroid = np.mean(queryEmbeddings[queryLabels == label, :], axis = 0)
        queryCentroids = np.vstack((queryCentroids, centroid))
    
    # Init knn
    knn = KNeighborsClassifier(refCentroids.shape[0], metric = "l2")
    knn.fit(refCentroids, refUniqueLabels)
    # Get neighbors    
    (dis, neighbors) = knn.kneighbors(queryCentroids, return_distance=True)
    
    # Get matches
    matches = {}
    for i, queryLabel in enumerate(queryUniqueLabels):
        if dis[i, 0] < disThreshold:
            matches[queryLabel] = queryUniqueLabels[neighbors[i,0]]
        else:
            matches[queryLabel] = currentId
            currentId += 1
    
    # Create net tracksEmbeddings
    resultTracksEmbeddings = copy.deepcopy(tracksEmbeddings1)
    
    for queryRealId in matches.keys():
        queryNewId = matches[queryRealId]
        
        queryNewLabels = [queryNewId]*np.sum(queryLabels == queryRealId)
        resultTracksEmbeddings["ids"] = resultTracksEmbeddings["ids"] + queryNewLabels
        resultTracksEmbeddings["embeddings"] = np.vstack((resultTracksEmbeddings["embeddings"], queryEmbeddings[queryLabels == queryRealId, :]))
    
    return matches, resultTracksEmbeddings

def ReIdByVoting(tracksEmbeddings1, tracksEmbeddings2, disThreshold):
    """
    This function giving the track information of 2 multi-object single-camera trackings results
    re-assigns ids matching different camera tracks. For the matching voting method is used.

    Parameters
    ----------
    tracksEmbeddings1 : dict
        Reference tracking results.
    tracksEmbeddings2 : dict
        Query tracking results.
    disThreshold : float
        Embedding distance threshold to be considered same track.

    Returns
    -------
    matches : dict
        Assignmests of old query labels and new ones.
    resultTracksEmbeddings : dict
        Updated results with 2 camera results.

    """
    
    refLabels = np.array(tracksEmbeddings1["ids"]) 
    refUniqueLabels = np.unique(refLabels)
    refEmbeddings = tracksEmbeddings1["embeddings"]
    
    currentId = np.sort(refUniqueLabels)[-1] + 1
    
    queryLabels = np.array(tracksEmbeddings2["ids"]) 
    queryUniqueLabels = np.unique(queryLabels)
    queryEmbeddings = tracksEmbeddings2["embeddings"]
    
    # Init knn
    knn = KNeighborsClassifier(1, metric = "l2")
    knn.fit(refEmbeddings, refLabels)
    # Get neighbors    
    (dis, neighbors) = knn.kneighbors(queryEmbeddings, return_distance=True)
    
    # Voting
    votes = np.zeros((len(queryUniqueLabels), len(refUniqueLabels)))
    for i, queryLabel in queryLabels:
        predDis = dis[i, 0]
        
        if predDis > disThreshold:
            continue
        
        queryIndex = np.where(queryUniqueLabels == queryLabel)[0]
        refIndex = np.where(refUniqueLabels == refLabels[neighbors[i,0]])[0]
        votes[queryIndex, refIndex] += 1
    
    # Get matches
    matches = {}
    for i, queryLabel in enumerate(queryUniqueLabels):
        maxVotes = np.max(votes[i,:])
        maxVotesIndex = np.argmax(votes[i,:])
        
        if maxVotes > 0:
            matches[queryLabel] = queryUniqueLabels[maxVotesIndex]
        else:
            matches[queryLabel] = currentId
            currentId += 1
    
    # Create net tracksEmbeddings
    resultTracksEmbeddings = copy.deepcopy(tracksEmbeddings1)
    
    for queryRealId in matches.keys():
        queryNewId = matches[queryRealId]
        
        queryNewLabels = [queryNewId]*np.sum(queryLabels == queryRealId)
        resultTracksEmbeddings["ids"] = resultTracksEmbeddings["ids"] + queryNewLabels
        resultTracksEmbeddings["embeddings"] = np.vstack((resultTracksEmbeddings["embeddings"], queryEmbeddings[queryLabels == queryRealId, :]))
    
    return matches, resultTracksEmbeddings
