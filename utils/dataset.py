from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_data():
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=0.5, std=0.5)])
    
    training_set = datasets.StanfordCars(root = 'data', 
                                         split= 'train', 
                                         download= True, 
                                         transform=transform)
    
    training_loader = DataLoader(training_set, batch_size=64, shuffle=True, num_workers=4)

    test_set = datasets.StanfordCars(root = 'data', 
                                         split= 'test', 
                                         download= True, 
                                         transform=transform)
    
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)