import pickle

from RBM_model import *
from dataset import transform_test
from sklearn.metrics import f1_score

batch_size = 16
val_set = datasets.ImageFolder(root='../dataset_test/', transform=transform_test)
val_load = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False)
print("Loading testing set ....")

model = RBM(n_visible=784, n_hidden=384, steps=1, learning_rate=0.1, momentum=0, decay=0, temperature=1)

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model("./model/RBM_new.pk")

n_classes = 2

fc = nn.Linear(model.n_hidden, 2)
fc.load_state_dict(torch.load("./model/fc_new.pt"))
fc.eval()

from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Initialize variables for accuracy, F1 score, and confusion matrix
correct = 0
iterations = 0
y_true = []
y_pred = []

for x_batch, y_batch in tqdm(val_load):
    x_batch = x_batch.reshape(x_batch.size(0), model.n_visible)

    x_batch = x_batch
    y_batch = y_batch

    y = model(x_batch)
    y = fc(y)

    predicted = torch.argmax(y, dim=1)
    
    # Update variables for accuracy calculation
    correct += (predicted == y_batch).sum()
    iterations += 1
    
    # Update true and predicted labels for F1 score calculation
    y_true.extend(y_batch.tolist())
    y_pred.extend(predicted.tolist())

# Calculate accuracy
accuracy = round(((100 * correct) / len(val_set)).item(), 2)
print("Accuracy on test set: {}%".format(accuracy))

# Calculate F1 score
f1 = f1_score(y_true, y_pred)
print("F1 score on test set: {}".format(f1))

# Calculate confusion matrix
labels = np.unique(y_true)
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Normalize confusion matrix to percentages
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# Plot confusion matrix
fig, ax = plt.subplots()
im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
classes = ["Mask", "No Mask"]
ax.set(xticks=np.arange(cm_normalized.shape[1]),
       yticks=np.arange(cm_normalized.shape[0]),
       xticklabels=classes, yticklabels=classes,
       xlabel='Predicted label', ylabel='True label',
       title='Confusion Matrix (%)')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        ax.text(j, i, format(cm_normalized[i, j], '.2f') + '%',
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > cm_normalized.max() / 2. else "black")
plt.tight_layout()
plt.savefig("confusion.png", bbox_inches='tight')
plt.show()