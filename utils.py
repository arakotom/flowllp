
#%%

import math
import os
import time

import numpy as np
import ot 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_clf(clf, train_loader,weights=None,lr=0.001, num_epochs=1000,val_loader=None,test_loader=None,
              n_class=10):
        clf.train()  # Set the model to training mode

        clf.to(device)
        if weights is None:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(clf.parameters(), lr=lr,weight_decay=1e-4)
        max_bal_acc_val = 0
        min_error_val = 100000
        best_bal_acc_test = 0
        best_acc_test = 0
        # Training loop
        for epoch in range(num_epochs):
            #print(f'Epoch [{epoch + 1}/{num_epochs}]')
            loss_total = 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # Clear gradients
                outputs = clf(inputs.to(device))  # Forward pass
                loss = criterion(outputs, labels.to(device))  # Compute loss
                loss_total += loss.item()
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights


            if epoch % 10 == 0 and val_loader is not None:
                if isinstance(val_loader, np.ndarray):
                    bal_acc_val = accur_on_bag_prop(clf,val_loader,n_class=n_class)
                else:
                    acc_val, bal_acc_val, _ = evaluate_clf(clf, val_loader,n_classes=n_class)
                if bal_acc_val > max_bal_acc_val:
                    max_bal_acc_val = bal_acc_val
                    acc_test, bal_acc_test, _ = evaluate_clf(clf, test_loader,n_classes=n_class)
                    best_bal_acc_test = bal_acc_test
                    best_acc_test = acc_test    
 
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_total/len(train_loader):.4f}  {max_bal_acc_val:.4f} {best_bal_acc_test:.4f} ')
        return clf, max_bal_acc_val, best_acc_test, best_bal_acc_test


def train_clf_bags(clf, train_loader,bags, weights=None,lr=0.001, num_epochs=1000,val_loader=None,test_loader=None,
              n_class=10,lmbd_bag = 1,lmbd_train=1):
        clf.train()  # Set the model to training mode

        clf.to(device)
        if weights is None:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.Adam(clf.parameters(), lr=lr,weight_decay=1e-4,betas=(0.9, 0.999))
        max_bal_acc_val = 0
        best_bal_acc_test = 0
        best_acc_test = 0
        min_error_val = 100000
        # Training loop
        for epoch in range(num_epochs):
            #print(f'Epoch [{epoch + 1}/{num_epochs}]')
            bags = np.random.permutation(bags)
            loss_total = 0
            for inputs, labels in train_loader:
                i_bag = np.random.randint(0,len(bags))
                inputs_bag = bags[i_bag]['data']
                y_prop = bags[i_bag]['prop']
                y_prob_mb = torch.Tensor(bags[i_bag]['prop']).to(device)


                optimizer.zero_grad()  # Clear gradients
                outputs = clf(inputs.to(device))  # Forward pass
                loss = lmbd_train*criterion(outputs, labels.to(device))  # Compute loss
                #loss = 0
                outputs_bag = clf(inputs_bag.to(device)) 

                D_loss_prop = -torch.mean(torch.sum(y_prob_mb * torch.log(torch.mean(torch.softmax(outputs_bag, dim=1), dim=0) + 1e-7), dim=0))

                loss += lmbd_bag*D_loss_prop
                #loss += lmbd*torch.sum(torch.abs(outputs_bag - torch.Tensor(y_prop).to(device).long()))

                loss_total += loss.item()
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights


            if epoch % 10 == 0 and val_loader is not None:
                if isinstance(val_loader, np.ndarray):
                    bal_acc_val = accur_on_bag_prop(clf,val_loader,n_class=n_class)
                else:
                    acc_val, bal_acc_val, _ = evaluate_clf(clf, val_loader,n_classes=n_class)
                if bal_acc_val > max_bal_acc_val:
                    max_bal_acc_val = bal_acc_val
                    acc_test, bal_acc_test, _ = evaluate_clf(clf, test_loader,n_classes=n_class)
                    best_bal_acc_test = bal_acc_test
                    best_acc_test = acc_test    
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_total/len(train_loader):.4f}  {max_bal_acc_val:.4f} {best_bal_acc_test:.4f} ')
        return clf, max_bal_acc_val, best_acc_test, best_bal_acc_test
    




def evaluate_clf(clf, test_loader,n_classes=10,return_pred=False):
    clf.eval()  # Set the model to evaluation mode

    correct = 0
    total = 0
    all_correct = []
    all_predicted = []
    all_soft = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            test_outputs = clf(inputs.to(device)).cpu()
            test_soft = torch.softmax(test_outputs,1)
            _, predicted = torch.max(test_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_correct.append(labels)
            all_predicted.append(predicted)
            all_soft.append(test_soft)
    accuracy = correct / total
    #print(f'Accuracy: {accuracy:.4f}')
    all_correct = torch.cat(all_correct)
    all_predicted = torch.cat(all_predicted)
    all_soft = torch.cat(all_soft)
    confusion_matrix = torch.zeros(n_classes,n_classes)
    b_accuracy = balanced_accuracy_score(all_correct, all_predicted)
    for t, p in zip(all_correct, all_predicted):
        confusion_matrix[t, p] += 1
    #print(confusion_matrix)
    clf.train()
    if return_pred:
        return accuracy, b_accuracy,confusion_matrix, all_soft
    else:
        return accuracy, b_accuracy, confusion_matrix
    
def create_data(X_train, y_train):
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    return train_dataset

def get_weights(labels):
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    num_class = len(np.unique(labels))
    from collections import Counter
    # Compute class frequencies
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    try:
        class_weights = [total_samples / (class_counts[i] * len(class_counts)) for i in range(len(class_counts))]
    except:
        class_weights = [1/num_class for i in range(num_class)]        # Convert class weights to a tensor
    weights = torch.tensor(class_weights).to(device)
    return weights


def extract_data_label(Bag, type_label = 'label', type_data = 'data'):
    num_bags = len(Bag)
    
    X = torch.cat([Bag[i][type_data] for i in range(num_bags)],0)
    y = torch.cat([Bag[i][type_label] for i in range(num_bags)],0)
    if type_label == 'label' or type_label == 'y_pred':
        y = y.long()
    return X,y


def accur_on_bag_prop(model,val_bags,n_class=10):
    val_accur = []
    for bag in val_bags:
        data = bag['data']
        label = bag['label']
        prop = bag['prop']
        data = data.to(device).float()
        with torch.no_grad():
            outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total = data.size(0)
        prop_pred = torch.bincount(predicted,minlength=n_class).float()/total
        acc = torch.sum(torch.abs(prop_pred - torch.Tensor(prop).to(device))).item()
        val_accur.append(acc)
    return n_class - torch.Tensor(val_accur).mean().item()


# def error_bag(Bag, clf):
#     error = 0
#     for bag in Bag:
#         with torch.no_grad():
#             y_pred = clf(bag['data'])
#         y_pred = torch.softmax(y_pred,1).mean(0)
#         y_bag = torch.Tensor(bag['prop'])
#         error += torch.sum(torch.abs(y_pred - y_bag))
#     error /= len(Bag)
#     return error

def train_easyllp(model, Bag, type_data='data',n_class=10, num_epochs=1000, lr = 0.0001, 
                  verbose = True, val_loader= None,test_loader = None):

    model.train()  # Set the model to training mode
    model.to(device)
    list_accuracy = []
    criterion = nn.CrossEntropyLoss()
    max_bal_acc_val = 0
    min_error_val = 100000

    prop = torch.zeros(n_class)
    for i in range(len(Bag)):
        prop += torch.Tensor(Bag[i]['prop'])
    prop = prop/len(Bag)

    optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.9,weight_decay=1e-4)

    for epoch in range(num_epochs):
        #indices = torch.randperm(len(Bag))
        indices = np.random.permutation(len(Bag))

        for i in range(len(Bag)):
            B_i = Bag[indices[i]][type_data].to(device)
            alpha = Bag[indices[i]]['prop']
            #B_i = Bag[i][type_data].to(device)
            #alpha = Bag[i]['prop']
            k = B_i.shape[0]
            # Pick j from [k] uniformly at random
            j = torch.randint(0, len(B_i), (1,)).item()
            x_tj = B_i[j].to(device)
            if len(x_tj.shape) == 3:
                x_tj = x_tj.unsqueeze(0)
            # Forward pass
            y_pred = model(x_tj).reshape(1,-1)
            y_pred = torch.nn.functional.softmax(y_pred, dim=1)
            loss = 0
            for yc in range(n_class):
                lab = torch.Tensor([yc]).long().to(device)
                loss +=  (k*alpha[yc] - (k-1)*prop[yc])*criterion(y_pred, lab  )

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if verbose:
            #     print(epoch,loss.item())
        if epoch % 10 == 0 and val_loader is not None:
            if isinstance(val_loader, numpy.ndarray):
                error_val = error_bag(val_loader, model)
                if error_val < min_error_val:
                    min_error_val = error_val
                    acc_test, bal_acc_test, _ = evaluate_clf(model, test_loader,n_classes=n_class)
                    best_bal_acc_test = bal_acc_test
                    best_acc_test = acc_test
                    print(f'Accuracy on the ts: {acc_test:.4f} {bal_acc_test:.4f} at epoch {epoch}')


            else:
                acc_val, bal_acc_val, _ = evaluate_clf(model, val_loader,n_classes=n_class)
                print(f'Accuracy on the val: {acc_val:.4f} {bal_acc_val:.4f} at epoch {epoch}')
                if bal_acc_val > max_bal_acc_val:
                    max_bal_acc_val = bal_acc_val
                    acc_test, bal_acc_test, _ = evaluate_clf(model, test_loader,n_classes=n_class)
                    best_bal_acc_test = bal_acc_test
                    best_acc_test = acc_test
                    print(f'Accuracy on the ts: {acc_test:.4f} {bal_acc_test:.4f} at epoch {epoch}')

    if test_loader is not None:
        return model,max_bal_acc_val, best_acc_test, best_bal_acc_test
    else:
        return model

# %%
