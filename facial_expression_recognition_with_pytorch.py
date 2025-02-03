
import numpy as np
import matplotlib.pyplot as plt
import torch

# Configurations

TRAIN_IMG_FOLDER_PATH = '/content/Facial-Expression-Dataset/train/'
VALID_IMG_FOLDER_PATH = '/content/Facial-Expression-Dataset/validation/'
#LEARNING RATE
LR=0.001
BATCH_SIZE=32
EPOCHS=15
DEVICE='cuda'
MODEL_NAME='efficientnet_b0'

"""# Load Dataset"""

from torchvision.datasets import ImageFolder
from torchvision import transforms as T

train_augs=T.Compose([
    
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=(-20,+20)),
    T.ToTensor() 

])

valid_augs=T.Compose([
    T.ToTensor()
])

trainset=ImageFolder(TRAIN_IMG_FOLDER_PATH,transform=train_augs)
validset=ImageFolder(VALID_IMG_FOLDER_PATH,transform=valid_augs)

print(f"Total no. of examples in trainset : {len(trainset)}")
print(f"Total no. of examples in validset : {len(validset)}")

print(trainset.class_to_idx)

image,label = trainset[8090]
plt.imshow(image.permute(1,2,0)) #(h,w,c)
plt.title(label)
plt.show()

"""# Load Dataset into Batches"""

from torch.utils.data import DataLoader

trainloader = DataLoader(trainset , batch_size=BATCH_SIZE , shuffle=True)
validloader = DataLoader(validset,batch_size=BATCH_SIZE)

print(f"Total no. of batches in trainloader : {len(trainloader)}")
print(f"Total no. of batches in validloader : {len(validloader)}")

for images,labels in trainloader:
  break;



print(f"One image batch shape : {images.shape}")#32,3,48,48-> no. of images in 1 batch, channel size , image size(48,48)
print(f"One label batch shape : {labels.shape}")

"""# Create Model"""

import timm
from torch import nn

class FaceModel(nn.Module):
  def __init__(self):
    super(FaceModel , self).__init__()

    self.eff_net = timm.create_model('efficientnet_b0',pretrained=True,num_classes=7)


  def forward(self , images , labels = None):
    logits=self.eff_net(images) # without any sigmoid , or softmax applied on it on the final layer

    if labels!=None:
      loss=nn.CrossEntropyLoss()(logits,labels)
      return logits , loss

    return logits

model=FaceModel()
model.to(DEVICE)

"""# Create Train and Eval Function"""

from tqdm import tqdm
#tqdm is a Python library used to display progress bars in loops

def multiclass_accuracy(y_pred,y_true):
    top_p,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))

def train_fn(model,dataloader,optimizer,current_epo):
  model.train()
  total_loss=0
  total_acc=0
  tk=tqdm(dataloader , desc="EPOCH"+"[TRAIN]"+str(current_epo+1)+"/"+str(EPOCHS))
  #This line creates a progress bar using tqdm to track the progress of the training loop over the dataloader.


  for t,data in enumerate(tk):
    images,labels = data
    images ,labels =images.to(DEVICE),labels.to(DEVICE) #Transfer to GPU
    #images, labels = images.to(DEVICE), labels.to(DEVICE): Moves both the images and labels to the device specified by DEVICE (usually a GPU if available).

    optimizer.zero_grad()
    #PyTorch accumulates gradients by default.

    logits,loss = model(images,labels)
    loss.backward()
    #This updates the model's parameters based on the gradients computed during loss.backward().

    optimizer.step()

    total_loss+=loss.item()
    total_acc+=multiclass_accuracy(logits,labels)
    tk.set_postfix({'loss':'%6f' %float(total_loss / (t+1)),
                    'acc':'%6f' %float(total_acc / (t+1))})

  return total_loss/len(dataloader),total_acc/len(dataloader)

def eval_fn(model,dataloader,current_epo):
  model.eval()
  total_loss=0
  total_acc=0
  tk=tqdm(dataloader , desc="EPOCH"+"[VALID]"+str(current_epo+1)+"/"+str(EPOCHS))

  for t,data in enumerate(tk):
    images,labels = data
    images ,labels =images.to(DEVICE),labels.to(DEVICE) #Transfer to GPU


    logits,loss = model(images,labels)


    total_loss+=loss.item()
    total_acc+=multiclass_accuracy(logits,labels)
    tk.set_postfix({'loss':'%6f' %float(total_loss / (t+1)),
                    'acc':'%6f' %float(total_acc / (t+1))})

  return total_loss/len(dataloader),total_acc/len(dataloader)

"""# Create Training Loop"""

optimizer = torch.optim.Adam(model.parameters(),lr=LR)

import numpy as np
best_valid_loss=np.Inf


for i in range(EPOCHS):
  train_loss,train_acc = train_fn(model,trainloader,optimizer,i) # i is curent EPOCH
  valid_loss,valid_acc = eval_fn(model,validloader,i)

  if valid_loss<best_valid_loss:
    best_valid_loss = valid_loss
    torch.save(model.state_dict(),"best-weights.pt")
    print("Saved Best wiights")
    best_valid_loss = valid_loss

#Model's Prediction: The model generates predictions (logits or probabilities).
#Loss Calculation: The predicted values are compared to the true values using the loss function.
#Loss Feedback: The loss is used to compute the gradients, which tell us how to adjust the weights to improve the model’s predictions.
# Parameter Update: The optimizer uses the gradients to update the model’s parameters (weights and biases) to minimize the loss.

"""# Inference"""

import torch
from torchvision import transforms as T
from PIL import Image


model = FaceModel()
model.load_state_dict(torch.load('/content/best-weights.pt'))
model.eval()
model.to(DEVICE)


inference_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

def predict_image(image_path):
    # image = Image.open('/content/happy.jpg')
    # image = Image.open('/content/happy.jpg')
    image = Image.open('/content/sad.jpg')
    image = inference_transforms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()


    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    predicted_class_name = class_names[predicted_class]

    return predicted_class_name, probabilities


image_path = 'img_path'
predicted_label, probabilities = predict_image(image_path)

print(f"Predicted Class: {predicted_label}")
print(f"Probabilities: {probabilities}")

