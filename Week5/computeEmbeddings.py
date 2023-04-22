import os
from siamese_network import SiameseNetwork
import torchvision.transforms as transforms
import torch
from reId import obtainDetEmbeddings, saveAllEmbeddingsAndIds


if __name__ == "__main__":
    
    outputEmbeddings = "./S03embeddings/"

    if not os.path.exists(outputEmbeddings):
       os.makedirs(outputEmbeddings)
       
    camsPath = "../seqs/train/S03/"
    tracksPath = "./SEQ3_tracks_pp/"
    embeddingSize = 512
    weightsPath = "sm_constrastive_car.pt"    
    model = SiameseNetwork()
    model = model.to("cuda")
    model.load_state_dict(torch.load(weightsPath, map_location = "cuda"))
    
    for trackFile in os.listdir(tracksPath):
        print(trackFile)
        trackPath = tracksPath + trackFile
        
        videoPath = camsPath + trackFile[:-4].split("_")[-1] + "/vdo.avi"
    
    
    
        transform=transforms.Compose([transforms.Resize(100),
                                      transforms.RandomResizedCrop(100),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
     
        tracksEmbeddings = obtainDetEmbeddings(model, transform, trackPath, videoPath, embeddingSize)
    
        saveAllEmbeddingsAndIds(tracksEmbeddings, outputEmbeddings + trackFile[:-4])
