#%%
import math
import os
import time
from ot.da import sinkhorn_lpl1_mm, emd_laplace

import numpy as np
import ot 
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data import create_toy_bags
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from ot.utils import (
    dist,
    kernel,
    laplacian,
    dots,
)
import copy

from models import Generator, Discriminator, FullyConnectedClassifier   
from utils import evaluate_clf    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils import accur_on_bag_prop

def entropy_loss(v):
    """
    Entropy loss for probabilistic prediction vectors
    """
    return torch.mean(torch.sum(- torch.softmax(v, dim=1) * torch.log_softmax(v, dim=1), 1))




def train_bagloss(Bag, D, n_class,num_epochs,lr=0.0001,val_loader= None,
                 test_loader=None,
                 val_bags=None):
    D = D.to(device)
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    max_bal_acc_val = 0
    best_bal_acc_test = 0
    best_acc_test = 0
    bal_acc_val = 0
    best_clf = None
    for epoch in range(num_epochs):
        timer = time.time()
        Bag = np.random.permutation(Bag)
        tot_bag_loss = 0    
        for mb in range(len(Bag)):
            X_mb = Bag[mb]['data']
            Y_mb = Bag[mb]['label']
            X_mb, Y_mb = X_mb.to(device), Y_mb.to(device)
            y_prob_mb = torch.Tensor(Bag[mb]['prop']).to(device)
            D.train()
            D.requires_grad_(True)
            optimizer_D.zero_grad()
            d_net_real = D(X_mb) 
            D_loss_prop = -torch.mean(torch.sum(y_prob_mb * torch.log(torch.mean(torch.softmax(d_net_real, dim=1), dim=0) + 1e-7), dim=0))

            D_loss = D_loss_prop #+ ent_loss
            D_loss.backward()
            optimizer_D.step()
            tot_bag_loss += D_loss_prop.item()/len(Bag)

        if epoch % 10 == 0 and val_loader is not None and test_loader is not None:
            D.eval()
            acc_val, bal_acc_val, _ = evaluate_clf(D, val_loader,n_classes=n_class)
            print(f'Accuracy on the val: {bal_acc_val:.4f} at epoch {epoch}')
            if bal_acc_val > max_bal_acc_val:
                max_bal_acc_val = bal_acc_val
                acc_test, bal_acc_test, _ = evaluate_clf(D, test_loader,n_classes=n_class)
                best_bal_acc_test = bal_acc_test
                best_acc_test = acc_test
                print(f'Accuracy on the ts: {bal_acc_val:.4f} {bal_acc_test:.4f} at epoch {epoch}')
                best_clf = copy.deepcopy(D)
                best_clf = best_clf.to(device)
                best_clf.eval()

        print(f'Epoch: {epoch}, Time: {time.time()-timer:.2f}, BagLoss {tot_bag_loss:2.4f} Val Accur: {bal_acc_val:.4f}'
              f' Test Accur: {best_bal_acc_test:.4f}')

    return D, max_bal_acc_val,best_acc_test, best_bal_acc_test, best_clf



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
    all_soft = torch.cat(all_soft,dim=0)
    confusion_matrix = torch.zeros(n_classes,n_classes)
    b_accuracy = balanced_accuracy_score(all_correct, all_predicted)
    for t, p in zip(all_correct, all_predicted):
        confusion_matrix[t, p] += 1
    #print(confusion_matrix)
    clf.train()
    if return_pred:
        return accuracy, b_accuracy,confusion_matrix, all_soft,all_correct
    else:
        return accuracy, b_accuracy, confusion_matrix


def bagloss_ensemble(Bag, clf_list, ratio=0.8, n_iter=10,n_class=3,num_epochs=100,lr=0.0001, val_loader= None,test_loader=None,
            val_bags=None):

    list_best_clf = []
    list_acc = []
    bc, bc_val = 0, 0
    for i in range(n_iter):
        Bag = np.random.permutation(Bag)
        n_bagging = int(len(Bag)*ratio)
        if n_bagging == len(Bag) or n_bagging == 0:
            n_bagging = len(Bag)-1

        Bag_bagging = Bag[:n_bagging]


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clf = clf_list[i].to(device)
        clf, max_bal_acc_val,best_acc_test, best_bal_acc_test, best_clf = train_bagloss(Bag_bagging, clf, n_class=n_class, 
                                                                        num_epochs=num_epochs,lr=lr,val_loader=val_loader,
                            test_loader=test_loader,val_bags=val_loader)
        list_best_clf.append(best_clf)
        list_acc.append(best_bal_acc_test)

    if val_loader is not None and test_loader is not None:
        for i in range(len(list_best_clf)):
            clf = list_best_clf[i]
            if i == 0: 
                eval_acc, bal_acc_val, cm, y_pred,y_true = evaluate_clf(clf, test_loader,n_classes=n_class,return_pred=True)
                eval_acc, bal_acc_val, cm, y_pred_val,y_true_val = evaluate_clf(clf, val_loader,n_classes=n_class,return_pred=True)
            
            else:
                eval_acc, bal_acc_val, cm, y_pred_aux,y_true = evaluate_clf(clf, test_loader,n_classes=n_class,return_pred=True)
                eval_acc, bal_acc_val, cm, y_pred_aux_val,y_true_val = evaluate_clf(clf, val_loader,n_classes=n_class,return_pred=True)
                y_pred_val += y_pred_aux_val
                y_pred += y_pred_aux
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred_val = torch.argmax(y_pred_val, dim=1)
    bc = balanced_accuracy_score(y_true, y_pred)
    bc_val = balanced_accuracy_score(y_true_val, y_pred_val)
    out = {
        'y_pred': y_pred,
        'y_true': y_true,
        'y_pred_val': y_pred_val,
        'y_true_val': y_true_val,
        'bc': bc,
        'bc_val': bc_val,
        'list_acc': list_acc,
        'list_best_clf': list_best_clf,
    }
    return out

if __name__ == '__main__':

    from data import create_bags_from_tabular, create_cifar_orig_bags
    from utils import extract_data_label
    import time

    # seed = 0
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    perf_individual = []
    perf_bagging = []
    perf_full = []
    for i in range(10):

        torch.manual_seed(i)
        np.random.seed(i)
        torch.cuda.manual_seed_all(i)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        list_acc = []



        num_bags = 0
        nb_class_in_bag = 100
        bag_size = 50
        data = 'dry_beans'
        n_class = 7
        Bag, val_loader, test_loader = create_bags_from_tabular(data, nb_class_in_bag= nb_class_in_bag,
                                                                            bag_size=bag_size,
                                                                            dep_sample=1)
        

        dim = Bag[0]['data'].shape[1]




        lr = 0.001
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        D = FullyConnectedClassifier(dim,n_class).to(device)
        D, max_bal_acc_val,best_acc_test, best_bal_acc_test, best_clf = train_bagloss(Bag, D, n_class=n_class, 
                                                                        num_epochs=100,lr=lr,val_loader=val_loader,
                            test_loader=test_loader,val_bags=val_loader)
        print(f'Final Full error: {best_bal_acc_test:.4f}')
        perf_full.append(best_bal_acc_test)




        D = FullyConnectedClassifier(dim,n_class).to(device)
        n_iter = 20
        clf_list = [FullyConnectedClassifier(dim,n_class).to(device) for i in range(n_iter)]
        out = bagloss_ensemble(Bag, clf_list, n_iter=n_iter, ratio=0.8, n_class=n_class,num_epochs=100,
                                                             lr=lr,val_loader=val_loader,
                                                             test_loader=test_loader,val_bags=val_loader)
        
        bc = out['bc']
        bc_val = out['bc_val']
        list_acc = out['list_acc']
        print(f'Final Bagging error: {bc:.4f}')
        print(f'Final Bagging error val: {bc_val:.4f}')
        perf_bagging.append(bc)
        perf_individual.append(np.max(list_acc))
        

        

#
    print(f'Individual: {np.mean(perf_individual):.4f} Bagging: {np.mean(perf_bagging):.4f}')
    print(f'Full: {np.mean(perf_full):.4f} Bagging: {np.mean(perf_bagging):.4f}')

    np.array(np.array(perf_bagging) - np.array(perf_full))

# %%
