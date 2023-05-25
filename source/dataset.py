import torchvision.transforms as transforms
import torchvision.datasets as datasets

#Transform and load dataset
#Argument training_set by RandomResizedCrop, RandomHorizontalFlip
#Normalize data: [-1, 1]
transform_train = transforms.Compose([transforms.Resize((28, 28)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.flatten())
                                    ])

transform_test = transforms.Compose([transforms.Resize((28, 28)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.flatten())
                                    ])

