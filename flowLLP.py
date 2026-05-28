
#%%
import copy

import numpy as np
import ot
import torch
import torch.nn as nn
import torch.optim as optim

from bagloss import train_bagloss
from models import FullyConnectedClassifier
from utils import evaluate_clf, create_data, train_clf_bags

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dist_torch(x1,x2):
    x1p = x1.pow(2).sum(1).unsqueeze(1)
    x2p = x2.pow(2).sum(1).unsqueeze(1)
    prod_x1x2 = torch.mm(x1,x2.t())
    distance = x1p.expand_as(prod_x1x2) + x2p.t().expand_as(prod_x1x2) -2*prod_x1x2
    return distance 




def learn_labelprop_anchor(Bag, n_per_class_anchor=100, num_epochs=5000, n_class=10, dim=2,
                           anchor_init=None,lr=0.01,method='emd',
                           batch_bag = 1,
                           reg_label=0.01,
                           clf_bag =  None, # classifier over bags to regularize anchor learning
                           reg_bagclf = 1, # regularization parameter for clf_bag
                           verbose=True,
                           get_list_anchor=False,
                           debug=False,
                           data_shape = None):

    device = 'cpu'
    n_2 = n_per_class_anchor* n_class
    embedding = nn.Embedding(n_2, dim)
    if anchor_init is not None:
        print('Using anchor init')  
        if isinstance(anchor_init,torch.Tensor):
            embedding.weight.data = anchor_init
        else:
            embedding.weight.data = torch.from_numpy(anchor_init).float()
    else:
        embedding.weight.data = torch.randn(n_2,dim)*2
    
    ind_anchor = {}
    for i in range(n_class):
        ind_anchor[i] = np.arange(n_2//n_class*i,n_2//n_class*(i+1))
    y_anchor = np.array([i for i in range(n_class) for j in range(n_per_class_anchor)])
    y_anchor = torch.from_numpy(y_anchor).long().to(device)


    optimizer = optim.Adam(embedding.parameters(),lr = lr,betas=(0.9, 0.999))
    if get_list_anchor:
        if anchor_init is None:
            list_anchor = [embedding.weight.data.clone().cpu().numpy()]
        elif isinstance(anchor_init, torch.Tensor):
            list_anchor = [anchor_init.clone().cpu().numpy()]
        else:
            list_anchor = [torch.from_numpy(anchor_init).float().cpu().numpy()]
    list_nb_correct_match = []

    if clf_bag is not None:
        clf_bag.eval()
        clf_bag.to(device)

    for it in range(num_epochs):

        cond_reg_label = reg_label > 0 and it > 0

        S = 0
        np.random.shuffle(Bag)
        for j in range(batch_bag):
            # batch over the bags
            X = Bag[j]['data'].reshape(-1,dim).to(device)
            y = Bag[j]['label']
            n_1 = X.shape[0]


            if cond_reg_label:
                y_pred = Bag[j]['y_pred_noisy'].to(device)
                M_label_dist = torch.mm(torch.nn.functional.one_hot(y_pred,n_class).float(), 
                                       torch.nn.functional.one_hot(y_anchor,n_class).float().t())
                M_label_dist = 1 - M_label_dist
                M_label_dist = M_label_dist.to(device)

            a = torch.from_numpy(ot.unif(n_1)).float()
            b = np.zeros(n_2)
            for i in range(n_class):
               b[ind_anchor[i]] = Bag[j]['prop'][i]/n_per_class_anchor
            # b = np.ones(n_2)/n_2   ; just to check that uniform distribution breaks the learning
            b = torch.from_numpy(b)
            b /= torch.sum(b)
            a /= torch.sum(a)
            b = b.float()
            a = a.float()


            if cond_reg_label :
                M = dist_torch(X,embedding.weight.to(device)) + M_label_dist*reg_label
            else:
                M = dist_torch(X,embedding.weight.to(device)    )
            with torch.no_grad():

    
                gamma = ot.emd(a,b,M)
                text_ = 'emd'
                if debug: 
                    # track matching
                    gamma_aux = gamma.clone()
                    ind_b = torch.where(b>0)[0]
                    ind_sample_match = torch.argmax(gamma[:,ind_b],0).tolist()
                    label_sample_match = y[ind_sample_match]
                    nb_correct_match = torch.sum(label_sample_match == y_anchor[ind_b]).item()
                    n_anchor_to_match = len(ind_b)
                    #print(f'Correct match: {nb_correct_match}/{n_anchor_to_match} ({nb_correct_match/n_anchor_to_match:.2f})')
                    list_nb_correct_match.append(nb_correct_match/n_anchor_to_match)    
            gamma = gamma.to(device)    
            S +=  torch.sum(M*gamma)
            if clf_bag is not None:
                if data_shape is not None:
                    X_anchor_reshaped = embedding.weight.to(device).reshape([embedding.weight.shape[0]] + list(data_shape))
                else:
                    X_anchor_reshaped = embedding.weight.to(device)
                y_anchor_pred = clf_bag(X_anchor_reshaped)
                loss_clf = nn.CrossEntropyLoss()(y_anchor_pred, y_anchor.to(device))
                S += loss_clf*reg_bagclf

        embedding.zero_grad()
        S.backward()
        optimizer.step()
        if it % 100 == 0 and it > 0 and verbose:
            if not debug:
                print(it,S.item(),text_)
            else:
                print(it,S.item(),text_,nb_correct_match/n_anchor_to_match)
        if (it % 100 == 0 or it == num_epochs) and it > 0 and get_list_anchor:
            list_anchor .append(embedding.weight.data.clone().cpu().numpy())
    anchor = embedding.weight.detach().cpu().numpy()

    if get_list_anchor and not debug:
        return anchor, ind_anchor, list_anchor
    if not get_list_anchor and debug:
        return anchor, ind_anchor, list_nb_correct_match
    if get_list_anchor and debug:
        return anchor, ind_anchor, list_anchor, list_nb_correct_match
    return anchor, ind_anchor



#%%

if __name__ == '__main__':

    from data import create_bags_from_tabular
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # creating the dataset of bags and the leanring parameters
    nb_class_in_bag = 100 # all class
    bag_size = 100


    # these parameters are examples. They are not the optimal ones for the best performance.
    # and the code do not consider hyperparameter validation. 

    data = 'dry_beans'
    n_class = 7
    num_epochs = 30
    num_epochs_prop = 1000
    lr_mlp = 0.0001
    lr_prop = 0.001
    batch_size = 8
    # anchor flow parameters
    n_per_class_anchor = 500 
    reg_bagclf = 1
    reg_label = 10
    

    from data import create_bags_from_tabular

    Bag, val_loader, test_loader = create_bags_from_tabular(data, nb_class_in_bag= nb_class_in_bag,
                                                                        bag_size=bag_size,dep_sample=0,)
        

    n_per_class_anchor = n_per_class_anchor
    dim = Bag[0]['data'].shape[1]

    #%%
    # training the instance-level classifier with bagloss 
    # this is the initialization step

    clf_lab = FullyConnectedClassifier(n_input = dim, n_output = n_class)
    
    
    clf_lab = train_bagloss(Bag, clf_lab, n_class=n_class, num_epochs=num_epochs, lr=lr_mlp, val_loader=None,
                 test_loader=test_loader)[0]
    
    # assign noisy labels to samples in bags 
    # these labels are used in 
    for bag in Bag:
        X_bag = bag['data'].to(device)
        with torch.no_grad():
            y_pred_bag = clf_lab(X_bag)
            y_pred_bag = torch.argmax(y_pred_bag,1)
        bag['y_pred_noisy'] = y_pred_bag    
    acc, bal_acc_lab_init, cm = evaluate_clf(clf_lab, test_loader,n_classes=n_class)
    print('Performance of the initial classifier:', bal_acc_lab_init)

    #%%
    anchor_init = None


    # learning the anchor points 
    #

    shap = Bag[0]['data'].shape
    dim = np.prod(shap[1:])

    print('Learning anchor points with particle flow..')
    anchor, ind_anchor = learn_labelprop_anchor(Bag, n_per_class_anchor=n_per_class_anchor, 
                                                            num_epochs=num_epochs_prop, n_class=n_class,
                                                dim=dim, anchor_init=anchor_init,lr=lr_prop,
                                                method='emd',batch_bag=1,
                                                reg_label=reg_label,
                                                get_list_anchor=False,
                                                debug=False,
                                                clf_bag=clf_lab,
                                                reg_bagclf=reg_bagclf,
                                                data_shape= list(shap[1:]) if len(shap)>2 else None)                                      


    X_anchor = anchor
    y_anchor = np.zeros(anchor.shape[0])
    for i in range(n_class):
        y_anchor[ind_anchor[i]] = i
    X_anchor = torch.from_numpy(X_anchor)
    y_anchor = torch.from_numpy(y_anchor).long()
    train_ot_dataset = create_data(X_anchor,y_anchor)


    train_ot_loader = torch.utils.data.DataLoader(train_ot_dataset, batch_size=batch_size, shuffle=True)  

    # fine-tuning the instance-level classifier with learned anchor points and bag-level labels

    lmbd_bag = 1
    lmbd_train = 0.001
    num_epochs = 50

    clf_lab_i = copy.deepcopy(clf_lab)
    clf_lab_i = train_clf_bags(clf_lab_i, train_ot_loader, Bag, 
                             num_epochs=num_epochs,lr=lr_mlp,val_loader=None,lmbd_bag=lmbd_bag,
                             lmbd_train=lmbd_train)[0]

    acc, bal_acc_lab, cm = evaluate_clf(clf_lab_i, test_loader,n_classes=n_class)
    print(data,'Performance with learned anchors:',bal_acc_lab, lmbd_bag, lmbd_train)



#%%
