import pickle
import cv2
import argparse

from PIL import Image
from RBM_model import *

parser = argparse.ArgumentParser()

parser.add_argument('--path', help='Image and video link')

args = parser.parse_args() 
path = args.path

label2id = {
    0: 'Mask',
    1: 'No Mask'
}

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

model = RBM(n_visible=784, n_hidden=384, steps=1, learning_rate=0.1, momentum=0, decay=0, temperature=1)
model = load_model("./model/RBM1.pk")

fc = nn.Linear(model.n_hidden, 2)
fc.load_state_dict(torch.load("./model/fc1.pt"))

def putface(img, face, x, y, w, h):
    if(face == 'No Mask'):
        cv2.putText(img, face, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.rectangle(img, (x, y-5), (x+w, y+h), (0,0,255), 2)
    elif(face == 'Mask'):
        cv2.putText(img, face, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.rectangle(img, (x, y-5), (x+w, y+h), (0,255,0), 2)

def predict(img):
    transformx = transforms.Compose([transforms.Resize((28, 28)),
                                    transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.flatten())
                                    ]) 
    Image.fromarray(img).save("face.jpg")                            
    img = transformx(Image.fromarray(img))  
    # img = torch.unsqueeze(img, 0).to('cuda').float()    
 
    y = model(img)
    pred = fc(y)
    # print(pred)
    output_probs = F.softmax(pred, dim=0)  # apply softmax along dimension 1

    # Get the index of the highest probability class
    predicted_class = torch.argmax(output_probs)
    return predicted_class.item(), round(output_probs[predicted_class.item()].item(), 2)


#Load haarlike feature
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_frontalface_default.xml')

#Detect face
img = cv2.imread(path)
imgx = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

for (x,y,w,h) in faces:
    img2 = imgx[y+2:y+h-2, x+2:x+w-2]
    emo, _ = predict(img2)  #face index 
    face = label2id[emo]
    putface(img, face, x, y, w, h)

cv2.imwrite("Result.jpg", img)
print("Done predict, image has been saved: Result.jpg")