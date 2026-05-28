
#%%
import argparse
import os
import time
from torchvision import transforms
from torch.utils.data import Subset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
import copy

from utils import evaluate_clf, create_data
from models import FullyConnectedClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def create_bags_from_data_dep(train_labels, num_bags, nb_class_in_bag,embeddings=None, max_sample_per_bag=1e6):
 
    if isinstance(train_labels,np.ndarray):
        train_labels = torch.from_numpy(train_labels).long()

    n_class = len(torch.unique(train_labels))



    good_bag = False
    while not good_bag:
        Bag = []
        nb_time_class_is_inbag = np.zeros(n_class)
        for i in range(num_bags):
            class_in_bag = np.random.choice(n_class,nb_class_in_bag,replace=False)
            Bag.append({'class':class_in_bag})
            nb_time_class_is_inbag[class_in_bag] += 1
        if np.all(nb_time_class_is_inbag>0):
            good_bag = True

    print('Repartition of class accross bag', nb_time_class_is_inbag)

    # separating each class of the dataset into subsets according to the number of time they are in  bags
    class_for_bag = []
    for i_cls in range(n_class):
        ind = torch.where(train_labels==i_cls)[0].numpy()
        n_i_class = len(ind)

        permuted_array = np.random.permutation(ind)
        nb_subset = nb_time_class_is_inbag[i_cls].astype(int)
        # Step 2: Split the permuted array into k subsets
        ind_subset = np.sort(np.random.choice(n_i_class, nb_subset - 1, replace=False))

        start_subset =  np.concatenate(([0], ind_subset))
        end_subset = np.concatenate((ind_subset, [n_i_class]))
        subsets = [permuted_array[start_subset[i]:end_subset[i]] for i in range(nb_subset)]
        class_for_bag.append(subsets)
        #print(i_cls,subsets)
    # assign the data to the bag
    for i_bag in range(num_bags):
        class_in_bag = Bag[i_bag]['class']
        ind_for_bag = []
        for i_cls in class_in_bag:
            if len(class_for_bag[i_cls]) == 0:
                print('Error: not enough data for class',i_cls)
            ind_class = class_for_bag[i_cls].pop()
            Bag[i_bag][i_cls] = ind_class
            ind_for_bag.extend(ind_class)

        if len(ind_for_bag) > max_sample_per_bag:
            ind_for_bag = np.random.choice(ind_for_bag, max_sample_per_bag, replace=False)

        Bag[i_bag]['ind'] = ind_for_bag
        #Bag[i_bag]['data'] = train_data[ind_for_bag]
        Bag[i_bag]['label'] = train_labels[ind_for_bag]
        if embeddings is not None:
            Bag[i_bag]['embeddings'] = embeddings['train'][ind_for_bag]


        # computing propotion of each class in the bag
        #print(i_bag,class_in_bag,ind_for_bag)
        prop = [torch.sum(train_labels[ind_for_bag]==i_cls).item()/len(ind_for_bag) for i_cls in range(n_class)]
        Bag[i_bag]['prop'] = prop

    return Bag
class IndexDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx]

def get_batch_by_indices(dataset, indices):
    batch = [dataset[i] for i in indices]
    X_ = torch.stack([item[0] for item in batch])
    Y_ = torch.tensor([item[1] for item in batch])
    return X_, Y_

class FeatureLogitsWrapper(nn.Module):
    def __init__(self, feature_extractor, projector, classifier, num_classes=10):
        super(FeatureLogitsWrapper, self).__init__()
        self.feature_extractor = feature_extractor
        self.projector = projector
        self.classifier = classifier
        # Load the pre-trained ResNet18

        # Split the model into feature extractor and final layer
        self.features = nn.Sequential(feature_extractor,projector)
    def forward(self, x, return_features=False):
        # Extract features
        features = self.features(x)  # Shape: (batch_size, 512, 1, 1)
        features_flat = features.view(features.size(0), -1)  # Flatten to (batch_size, 512)

        # Compute logits
        logits = self.classifier(features_flat)  # Shape: (batch_size, num_classes)

        if return_features:
            return features_flat, logits
        else:
            return logits

if __name__ == '__main__':

    #sys.argv =['']

    # create argparse to get parameters from command line for all parameters
    parser = argparse.ArgumentParser(description='Parameters for training')
    parser.add_argument('--data', type=str, default='cifar10', help='Dataset to use (mnist, cifar10, etc.)')
    parser.add_argument('--nb_class_in_bag', type=int, default=10, help='Number of classes in each bag')
    parser.add_argument('--bag_size', type=int, default=50, help='Size of each bag')
    parser.add_argument('--img_size', type=int, default=64, help='Image size for resizing')
    parser.add_argument('--latent_dim', type=int, default=50, help='Latent dimension size')
    parser.add_argument('--n_class', type=int, default=10, help='Number of classes')

    # parameters for bag loss training
    parser.add_argument("--train_bagloss", action="store_true", help="Reserved compatibility flag (currently unused).")
    parser.add_argument('--num_epochs_bagloss', type=int, default=50, help='Number of epochs for bag loss training')
    parser.add_argument('--lr_bagloss', type=float, default=0.0001, help='Learning rate for bag-loss pretraining')
    
    # parameters for anchor learning
    parser.add_argument("--train_anchor", action="store_true", help="Reserved compatibility flag (currently unused).")
    parser.add_argument('--n_per_class_anchor', type=int, default=1000, help='Number of anchors per class')
    parser.add_argument('--num_epochs_prop', type=int, default=3000, help='Number of epochs for propagation training')
    parser.add_argument('--lr_prop', type=float, default=0.001, help='Learning rate for propagation training')
    parser.add_argument('--reg_label', type=float, default=0.0, help='Regularization for label propagation')
    parser.add_argument('--reg_bagclf', type=float, default=1.0, help='Regularization for bag classifier')
    
    # parameters for final training
    parser.add_argument('--num_epochs_all', type=int, default=100, help='Number of epochs for all training')
    parser.add_argument('--lr_all', type=float, default=0.0001, help='Learning rate for all training')
    parser.add_argument('--lmbd_bagloss', type=float, default=1, help='Weight for bag-proportion loss in final training')
    parser.add_argument('--lmbd_anchor', type=float, default=0.1, help='Weight for anchor supervised loss in final training')
    parser.add_argument('--outer_iter', type=int, default=1, help='Number of outer iterations')
    parser.add_argument('--single_lmbd_anchor', action="store_true", help="Reserved compatibility flag (currently unused).")

    parser.add_argument('--s', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    # args.num_epochs_bagloss = 1
    # args.num_epochs_prop = 2
    # args.num_epochs_all = 2


    seed = args.s
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    list_lmbd_anchor = [args.lmbd_anchor]
    savedir = f"./results/end2end/{args.data}_{args.img_size}/single/"

    savedir_model = f'./models/{args.data}_{args.img_size}/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)    
    if not os.path.exists(savedir_model):
        os.makedirs(savedir_model)
    filesave = ''
    results = {}
    for arg in vars(args):
        text_arg = arg
        text_arg = text_arg.replace('num_epochs','ne')
        text_arg = text_arg.replace('train_','tr_')
        text_arg = text_arg.replace('bagloss','bl')
        text_arg = text_arg.replace('latent_dim','ld')
        text_arg = text_arg.replace('anchor','anc')
        text_arg = text_arg.replace('nb_class_in_bag','in_bag')
        text_arg = text_arg.replace('outer_iter','oi')
        text_arg = text_arg.replace('single_lmbd_anc','sla')


        if filesave == '':
            filesave += f"{text_arg}-{getattr(args, arg)}"
        else:
            filesave += f"-{text_arg}-{getattr(args, arg)}"
    print(len(filesave),filesave)
    print('Results will be saved in',os.path.join(savedir, filesave + '.npz'))
    print('List_lmbd_anchor:', list_lmbd_anchor)
    print(len(filesave))
    #%%
    data = args.data

    nb_class_in_bag = args.nb_class_in_bag

    bag_size = args.bag_size
    IMG_SIZE = args.img_size
    if data == 'mnist':
        transform = transforms.Compose([
                                    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize images to 224x224
                                    transforms.Lambda(lambda x: x.convert("RGB")),  # Convert to 3 channels by duplicating grayscale
                                    transforms.ToTensor(),  # Convert images to PyTorch tensors
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3 channels
                                    ])



        # Load the training dataset
        full_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([
                                        transforms.Resize((IMG_SIZE, IMG_SIZE)),  # ResNet expects 224x224 input
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                        ])


        batch_size = 128
        # CIFAR-10 Dataset
        full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    #%%



    n_train = len(full_train_dataset)
    index = np.arange(n_train)
    np.random.shuffle(index)
    train_dataset = Subset(full_train_dataset, index[n_train//10:n_train])
    val_dataset = Subset(full_train_dataset, index[:n_train//10])
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=2,
                                               pin_memory=True,
                                               drop_last=True,
                                               )


    # extracting the train labels from the full dataset for creating bags
    if data == 'mnist':
        train_labels = full_train_dataset.targets[train_dataset.indices].numpy()
    else:
        train_labels = np.array(full_train_dataset.targets)[train_dataset.indices]




    num_bags = len(train_dataset)//bag_size
    Bag = create_bags_from_data_dep(train_labels, num_bags, nb_class_in_bag,embeddings=None, max_sample_per_bag=1e6)

    custom_dataset = IndexDataset(train_dataset)


    #%%
    #  Defining the model
    latent_dim = args.latent_dim
    n_class = args.n_class  

    #clf_lab = resnet18(pretrained=True)  # dowload pre-trained model


    import torchvision.models as models
    clf_lab = models.resnet18() #models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    checkpoint = torch.load('resnet18-f37072fd.pth', map_location='cpu')
    clf_lab.load_state_dict(checkpoint)




    feature_extractor = nn.Sequential(*list(clf_lab.children())[:-1],
                            nn.Flatten(),
                            )
    projector = nn.Sequential(
                nn.Linear(512,latent_dim,bias=True),
                nn.LeakyReLU(0.2),
                )
    classifier = FullyConnectedClassifier(n_input=latent_dim, n_output=n_class)
    clf_lab = nn.Sequential(
        feature_extractor,
        projector,
        classifier
    )
    # print all the arguments
    print('Arguments:')
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    #----------------------------------------------------------------------------
    #  Fine-tune the model with bag loss
    #----------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf_lab = clf_lab.to(device)
    lr_bagloss = args.lr_bagloss
    num_epochs_bagloss = args.num_epochs_bagloss

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf_lab.parameters(), lr=lr_bagloss, betas=(0.5, 0.999))

    for param in feature_extractor.parameters():
        param.requires_grad = False
    list_accuracy_bagloss = []
    print('Starting bag-loss pretraining...')
    max_bal_acc_val = 0
    for epoch in range(num_epochs_bagloss):
        clf_lab.train()
        timer = time.time()
        Bag = np.random.permutation(Bag)
        tot_bag_loss = 0    
        it = 0
        for mb in range(len(Bag)):

            indice = Bag[mb]['ind']


            X_mb, Y_mb = get_batch_by_indices(custom_dataset, indice)
            X_mb, Y_mb = X_mb.to(device), Y_mb.to(device)
            y_prob_mb = torch.Tensor(Bag[mb]['prop']).to(device)
            optimizer.zero_grad()

            d_net_real = clf_lab(X_mb) 
            D_loss_prop = -torch.mean(torch.sum(y_prob_mb * torch.log(torch.mean(torch.softmax(d_net_real, dim=1), dim=0) + 1e-7), dim=0))

            D_loss = D_loss_prop #+ ent_loss
            D_loss.backward()
            optimizer.step()
            tot_bag_loss += D_loss_prop.item()/len(Bag)
            it += 1
            if it % 100 == 0:
                print(f'Epoch: {epoch}, It: {it} Time: {time.time()-timer:.2f}, BagLoss {tot_bag_loss:2.4f} ')
                timer = time.time()
                tot_bag_loss = 0




        clf_lab.eval()
        acc_val , bal_acc_lab_val, cm = evaluate_clf(clf_lab, val_loader,n_classes=n_class)
        acc, bal_acc_lab, cm = evaluate_clf(clf_lab, test_loader,n_classes=n_class)
        print(f'Epoch: {epoch}, It: {it} Time: {time.time()-timer:.2f}, BagLoss {tot_bag_loss:2.4f} Val Bal Acc: {bal_acc_lab_val:2.4f} Test Bal Acc: {bal_acc_lab:2.4f} ')
        list_accuracy_bagloss.append([bal_acc_lab_val, bal_acc_lab])
        if bal_acc_lab_val > max_bal_acc_val:
            max_bal_acc_val = bal_acc_lab_val
        # save the best model
            torch.save(feature_extractor.state_dict(),os.path.join(savedir_model, f"feature_extractor_resnet18_bagloss_{IMG_SIZE}_seed{seed}.pth"))
            torch.save(projector.state_dict(),os.path.join(savedir_model, f"projector_resnet18_bagloss_{IMG_SIZE}_seed{seed}.pth"))
            torch.save(classifier.state_dict(),os.path.join(savedir_model, f"classifier_resnet18_bagloss_{IMG_SIZE}_seed{seed}.pth"))
        if epoch >= num_epochs_bagloss*0.75:
            for param in feature_extractor.parameters():
                param.requires_grad = True







    #%%

    model = FeatureLogitsWrapper(feature_extractor, projector, classifier, num_classes=n_class).to(device)
    model.eval()
    acc, bal_acc_lab_bagloss, cm = evaluate_clf(model, test_loader,n_classes=n_class)
    print(bal_acc_lab_bagloss)

    
    k = 0
    model.eval()
    model.to(device)
    print('Extracting features and noisy labels for bags...')
    for bag in Bag:
        indice = bag['ind']
        X_mb, Y_mb = get_batch_by_indices(custom_dataset, indice)
        X_mb = X_mb.to(device)
        with torch.no_grad():
            X_mb, y_pred = model(X_mb, return_features=True) 
        bag['data'] = X_mb.cpu()
        bag['y_pred_noisy'] = torch.argmax(y_pred,dim=1).cpu()
        k += 1

    #%%   
    from flowLLP import learn_labelprop_anchor

    anchor_init = None
    n_per_class_anchor = args.n_per_class_anchor
    num_epochs_prop = args.num_epochs_prop
    num_epochs_prop_init = num_epochs_prop 
    lr_prop = args.lr_prop
    reg_label = args.reg_label
    reg_bagclf = args.reg_bagclf
    dim = latent_dim
    list_outer_iter_results = []
    list_loss = []
    for it_outer in range(args.outer_iter):
        print(f'Starting outer-iter {it_outer}...')  
        clf_anchor =copy.deepcopy(classifier).to(device)
        feature_extractor.eval()
        projector.eval()
        clf_anchor.eval()
        print('Learning anchor model...')
        anchor, ind_anchor = learn_labelprop_anchor(Bag, n_per_class_anchor=n_per_class_anchor, 
                                                                num_epochs=num_epochs_prop, n_class=n_class,
                                                    dim=dim, anchor_init=anchor_init,lr=lr_prop,
                                                    method='emd',batch_bag=1,
                                                    reg_label=reg_label,
                                                    get_list_anchor=False,
                                                    debug=False,
                                                    clf_bag=clf_anchor,
                                                    reg_bagclf=reg_bagclf,
                                                )
        

            

        y_anchor = np.zeros(anchor.shape[0])
        for i in range(n_class):
            y_anchor[ind_anchor[i]] = i
        X_anchor = torch.from_numpy(anchor)
        y_anchor = torch.from_numpy(y_anchor).long()
        anchor_dataset = create_data(X_anchor,y_anchor)
        train_ot_loader = torch.utils.data.DataLoader(anchor_dataset, batch_size=64, shuffle=True)  
    


        #%%
        num_epochs_all = args.num_epochs_all
        lmbd_prop = args.lmbd_bagloss
        lmbd_anchor = args.lmbd_anchor

        print('Starting final training with bag loss and anchor supervision...')
        # we collect the results during training for all lmbd_anchor values  and validation/test accuracy
        list_results = []
        for lmbd_anchor in list_lmbd_anchor:
            # reload the bagloss pretrained model
            feature_extractor.load_state_dict(torch.load(os.path.join(savedir_model, f"feature_extractor_resnet18_bagloss_{IMG_SIZE}_seed{seed}.pth"),map_location=device))
            projector.load_state_dict(torch.load(os.path.join(savedir_model, f"projector_resnet18_bagloss_{IMG_SIZE}_seed{seed}.pth"),map_location=device))
            classifier.load_state_dict(torch.load(os.path.join(savedir_model, f"classifier_resnet18_bagloss_{IMG_SIZE}_seed{seed}.pth"),map_location=device))  
            print(f'lr_all: {args.lr_all}, num_epochs_all: {num_epochs_all}, lmbd_prop: {lmbd_prop}, lmbd_anchor: {lmbd_anchor}')
            clf_lab = nn.Sequential(feature_extractor, projector, classifier)
            clf_lab.train()

            optimizer = optim.Adam(clf_lab.parameters(), lr=args.lr_all, betas=(0.5, 0.999))

            for param in clf_lab.parameters():
                param.requires_grad = True
            clf_lab.to(device)
            max_bal_acc_val = 0
            for epoch in range(num_epochs_all):
                timer = time.time()
                Bag = np.random.permutation(Bag)
                tot_bag_loss = 0    
                it = 0
                
                for mb, (X_, y_) in enumerate(train_ot_loader):
                    indice = Bag[mb]['ind']
                    X_mb, Y_mb = get_batch_by_indices(custom_dataset, indice)
                    X_mb, Y_mb = X_mb.to(device), Y_mb.to(device)
                    y_prob_mb = torch.Tensor(Bag[mb]['prop']).to(device)
                    optimizer.zero_grad()
                    d_net_real = clf_lab(X_mb) 
                    D_loss_prop = -torch.mean(torch.sum(y_prob_mb * torch.log(torch.mean(torch.softmax(d_net_real, dim=1), dim=0) + 1e-7), dim=0))
                    y_pred_anchor = classifier(X_.to(device))
                    anchor_loss = nn.CrossEntropyLoss()(y_pred_anchor, y_.to(device))
                    D_loss = lmbd_prop*D_loss_prop + lmbd_anchor * anchor_loss
                    D_loss.backward()
                    optimizer.step()
                    tot_bag_loss += D_loss_prop.item()/len(Bag)
                    it += 1
                timer = time.time()
                list_loss.append(tot_bag_loss)

                clf_lab.eval()
                acc_val, bal_acc_lab_val, cm = evaluate_clf(clf_lab, val_loader,n_classes=n_class)
                #print('Validation balanced accuracy:', bal_acc_lab_val)
                acc, bal_acc_lab, cm = evaluate_clf(clf_lab, test_loader,n_classes=n_class)
                #print('Test balanced accuracy:', bal_acc_lab)
                if bal_acc_lab_val > max_bal_acc_val:
                    max_bal_acc_val = bal_acc_lab_val
                    clf_anchor_max = copy.deepcopy(classifier)

                print(f'Epoch: {epoch}, It: {it} Time: {time.time()-timer:.2f}, BagLoss {tot_bag_loss:2.4f} Val Bal Acc: {bal_acc_lab_val:2.4f} Test Bal Acc: {bal_acc_lab:2.4f} ')

                param = [lmbd_prop, lmbd_anchor, epoch, bal_acc_lab_val, bal_acc_lab]
                list_results.append(param)
                results = {'train_accuracy_bagloss': list_accuracy_bagloss,
                        'bal_acc_lab_bagloss': bal_acc_lab_bagloss,
                        'final_train_accuracy': np.array(list_results),
                        }
                np.savez(os.path.join(savedir, filesave + '.npz'), **results)
        #updating anchor for next outer-iter and classifier
        anchor_init = anchor.copy()
        if 'clf_anchor_max' not in locals():
            clf_anchor_max = copy.deepcopy(classifier)
        clf_anchor = clf_anchor_max
        clf_anchor.to(device)
        num_epochs_prop = min(1000, num_epochs_prop_init//3)
        list_outer_iter_results.append(list_results)
    multi_results = {'outer_iter_results': list_outer_iter_results,
                     'list_loss': np.array(list_loss),}
    final_results = dict(results)
    final_results.update(multi_results)
    np.savez(os.path.join(savedir, filesave + '.npz'), **final_results)

# %%
