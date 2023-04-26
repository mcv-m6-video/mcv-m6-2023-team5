import os
from siamese_network import SiameseNetwork
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
import torch
from reId import obtainDetEmbeddings, saveAllEmbeddingsAndIds


if __name__ == "__main__":
    
    outputEmbeddings = "./S04embeddings_resTrip1all/"

    if not os.path.exists(outputEmbeddings):
       os.makedirs(outputEmbeddings)
     
    resnet = True
    camsPath = "../seqs/train/S04/"
    tracksPath = "./S04_deepsortTracking_pp/"
    embeddingSize = 512
    trained = True
    weightsPath = "resnet18_sm_constrastive_3_car_lr_0.0001_batchSize_128_miner_no_triplet_1.pth"
    #weightsPath = "sm_constrastive_car.pt"
    if not resnet:    
        model = SiameseNetwork().eval()
    else:
        model1 = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        feature_extractor = torch.nn.Sequential(*(list(model1.children())[:-1]))
        model = torch.nn.Sequential(feature_extractor, torch.nn.Flatten()).eval()
    model = model.to("cuda")
    if trained:
        model.load_state_dict(torch.load(weightsPath, map_location = "cuda"))
    
    for trackFile in os.listdir(tracksPath):
        print(trackFile)
        trackPath = tracksPath + trackFile
        
        videoPath = camsPath + trackFile[:-4].split("_")[-1] + "/vdo.avi"
    
    
        if resnet:
            transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
        else:
            transform=transforms.Compose([transforms.Resize(100),
                                          transforms.RandomResizedCrop(100),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
     
        tracksEmbeddings = obtainDetEmbeddings(model, transform, trackPath, videoPath, embeddingSize, resnet)
    
        saveAllEmbeddingsAndIds(tracksEmbeddings, outputEmbeddings + trackFile[:-4])
