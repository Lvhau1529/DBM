import pickle

from RBM_model import *
from dataset import *

batch_size = 16
train_set = datasets.ImageFolder(root='../dataset_train/', transform=transform_train)
train_load = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
print('Train_set:', len(train_set))

print("Start training RBM model:")
model = RBM(n_visible=784, n_hidden=384, steps=1, learning_rate=0.1, momentum=0, decay=0, temperature=1)

# Trains an RBM
mse = model.fit(train_load, batch_size=16, epochs=32)

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

save_model(model, "./model/RBM_new.pk")

batch_size = 32
n_classes = 2
fine_tune_epochs = 16

fc = nn.Linear(model.n_hidden, 2)

# Cross-Entropy loss is used for the discriminative fine-tuning
criterion = torch.nn.CrossEntropyLoss()

# Creates the optimzers
optimizer = [optim.Adam(model.parameters(), lr=0.001),
             optim.Adam(fc.parameters(), lr=0.001)]

print("\n==============================")
print("Start training classify model:")
# For amount of fine-tuning epochs
for e in range(fine_tune_epochs):
    print(f'Epoch {e+1}/{fine_tune_epochs}')

    # Resetting metrics
    train_loss, val_acc = 0, 0
    iterations = 0
    correct = 0
    # For every possible batch
    for x_batch, y_batch in tqdm(train_load):
        for opt in optimizer:
            opt.zero_grad()
        
        x_batch = x_batch.reshape(x_batch.size(0), model.n_visible)

        x_batch = x_batch
        y_batch = y_batch

        y = model(x_batch)
        y = fc(y)
        
        loss = criterion(y, y_batch)
        
        loss.backward()
        
        for opt in optimizer:
            opt.step()

        predicted = torch.argmax(y, dim=1)
        correct += (predicted == y_batch).sum()
        iterations += 1

        train_loss += loss.item()
    acc = round(((100 * correct) / len(train_set)).item(),2)
    print(f'Loss: {train_loss / len(train_load)}, Acc: {acc}')

torch.save(fc.state_dict(), "./model/fc_new.pt")
print("Done training!!!")
