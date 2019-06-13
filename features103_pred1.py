import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import seaborn as sns
from torch.utils.data.sampler import SubsetRandomSampler
sns.set()
import sys
#import sklearn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


class test_data:
    def __init__(self, id, point, pred=0):
        self.id = id
        self.point = point
        self.pred = pred

class train_data:
	def __init__(self, id, point,label, pred=0):
		self.id = id
		self.point = point
		self.label = label
		self.pred = pred

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(103, 50)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = x.view(-1, 1*103) #One column representing all rgb bits
        x = F.sigmoid(self.fc1(x)) #Activation function
        x = self.fc1_drop(x)
        return F.log_softmax(self.fc2(x), dim=1)


def load_train_data():

    dict, data, testData, batchX, batchY = {}, [], [], [], []
    BATCH_SIZE = 829


    f = open("feature103_Train.txt")

    data = []
    floatData = []
    label = []
    id = []
    for lineNumber, line in enumerate(f):
        if lineNumber != 0:
            entries = line.split('\t')
            data.append(list(map(float, entries[2:])))
            label.append(float(entries[1]))
            id.append(entries[0])

    classLabel = np.array(label)
    trainData = np.array(data)
    #X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(trainData, classLabel, test_size = 0.33)
    f.close()

    dataTrain = []

    for idx in range(len(trainData)):
            # Create temp batch

        batchX.append(trainData[idx])

        batchY.append(classLabel[idx])

        if (idx + 1) % BATCH_SIZE == 0:

                # Create X and Y tuple and add to data

            x = np.array(batchX).reshape((BATCH_SIZE, 1, 103))
            y = np.array(batchY).reshape((BATCH_SIZE,))

            x = torch.from_numpy(x).float()
            y = torch.from_numpy(y).long()

            tuple = (x, y)

            dataTrain.append(tuple)

                # Reset batch
            batchX, batchY = [], []

    return dataTrain, id#, testData

def load_test_data():

    dict, data, testData, batchX, batchY = {}, [], [], [], []
    BATCH_SIZE = 20


    f = open("features103_test.txt")
    data = []
    id = []
    floatData = []
    label = []
    for lineNumber, line in enumerate(f):
        if lineNumber != 0:
            entries = line.split('\t')
            data.append(list(map(float, entries[1:])))
            id.append(entries[0])

    testData = np.array(data)

    f.close()

    dataTest = []

    for idx in range(len(testData)):
            # Create temp batch

        batchX.append(testData[idx])

        if (idx + 1) % BATCH_SIZE == 0:

                # Create X and Y tuple and add to data

            x = np.array(batchX).reshape((BATCH_SIZE, 1, 103))

            x = torch.from_numpy(x).float()
            dataTest.append(x)

                # Reset batch
            batchX, batchY = [], []


    return dataTest, id



def train(model, epoch, trainset_loader, device, optimizer, criterion, log_interval=1):
    # Set model to training mode
    model.train()

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(trainset_loader):
        # Copy data to GPU if needed
        #data = data.to(device)
        #target = target.to(device)

        # Zero gradient buffers

        optimizer.zero_grad()

        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = F.nll_loss(output, target)

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data), len(data)*33,
                100. * batch_idx / len(trainset_loader)))
    return float(loss.data.item())


def validate(loss_vector, accuracy_vector,model,testset_loader,device,criterion):
    model.eval()
    val_loss, correct = 0, 0
    prob_arr = []
    for data, target in testset_loader:
        #data = data.to(device)
        #target = target.to(device)
        output = model(data)

        for i in range(0,len(output)):
            prob_arr.append(torch.exp(output)[i])
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(testset_loader)
    #loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / 27357#len(testset_loader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, 27357, accuracy))
    return (float(accuracy)), prob_arr


def vtest(loss_vector, accuracy_vector,model,testset_loader,device,criterion):
    model.eval()
    #val_loss, correct = 0, 0
    preds = []
    for data in testset_loader:
        output = model(data)
        pred = torch.exp(output)
        for i in range(0,len(pred)):
            preds.append(pred[i])

    return (preds)

def singleTest(model,testpoint):
    model.eval()
    output = model(testpoint.point)
    pred = torch.exp(output)
    testpoint.pred = pred

def print_file(dat, name):
    fi = open(name,"w")
    for i in range(0,len(dat)):
        fi.write(str(dat[i].id) + "\t" + str(dat[i].pred[0][1].item()))
        fi.write("\n")
    fi.close()

def main():
    data,id_train=load_train_data()
    tester,id_test = load_test_data()

    unpackTest = []
    unpackTrainDat = []
    unpackTrainLab = []
    testing_mat = []
    training_mat = []

    for k in range(0,len(tester)):
        for i in range(0,len(tester[0])):
            unpackTest.append(tester[k][i])

    for j in range(0,len(unpackTest)):
        x = test_data(id_test[j],torch.tensor(unpackTest[j]))
        testing_mat.append(x)

    for i in range(0,len(data)):
        for j in range(0,len(data[0][0])):
            unpackTrainDat.append(data[i][0][j])
            unpackTrainLab.append(data[i][1][j])

    for i in range(0,len(id_train)):
        y = train_data(id_train[i],torch.tensor(unpackTrainDat[i]),unpackTrainLab[i])
        training_mat.append(y)

    learn_rate = .0001
    epochs = 15
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Using PyTorch version:', torch.__version__, ' Device:', device)

    model = Net1().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(learn_rate), momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    lossv, accv = [], []
    loss_arr = []
    valid_err = []
    prob_arr_epoch = []
    for epoch in range(1, epochs + 1):
        loss_arr.append(train(model,epoch,data,device,optimizer,criterion))
        prob_arr_epoch.append(vtest(lossv, accv,model,tester,device,criterion)) #Might only want to run this one time

    for i in range(0,len(testing_mat)):
        singleTest(model,testing_mat[i])

    for i in range(0,len(training_mat)):
        singleTest(model,training_mat[i])

    print_file(testing_mat,"features103_pred1.txt")
    #print_file(training_mat, "results_train103.txt")

if __name__ == "__main__":
    main()
