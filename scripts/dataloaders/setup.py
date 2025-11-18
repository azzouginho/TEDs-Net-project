
from torch.utils.data import DataLoader    


def setup_mnist_dataloader(params,subset_list):

    from dataloaders.mnist import MNIST_dataclass as MyDataset

    # --------- Dataloaders Functions:
    params_train = {'batch_size': params.batch, 'shuffle': True,
                    'num_workers': 8}

    params_val = {'batch_size': params.batch, 'shuffle': False,
                'num_workers': 8}  

    params_test = {'batch_size': params.batch, 'shuffle': False,
                'num_workers': 8}  


    dataloader_dict = {}
    if 'train' in subset_list:
        training_set = MyDataset(params,subset='Train')
        dataloader_dict['train'] = DataLoader(training_set, **params_train)
    if 'validation' in subset_list:
        val_set = MyDataset(params,subset='Validation')
        dataloader_dict['validation'] = DataLoader(val_set, **params_val)
    if 'test' in subset_list:
        test_set = MyDataset(params,subset='Test')
        dataloader_dict['test'] =DataLoader(test_set, **params_test)


    return dataloader_dict


def setup_acdc_dataloader(params,subset_list):    

    from dataloaders.ACDC import ACDC_dataclass as MyDataset

    # --------- Dataloaders Functions:
    params_train = {'batch_size': params.batch, 'shuffle': True,
                    'num_workers': 8}

    params_val = {'batch_size': params.batch, 'shuffle': False,
                'num_workers': 8}  

    params_test = {'batch_size': params.batch, 'shuffle': False,
                'num_workers': 8}  


    # Put in my dataloader functions:
    dataset_dict ="<AMEND WITH YOUR DATALOADER DICTIONARY>"

    # Into the torch dataloader function
    dataloader_dict = {}
    if 'train' in subset_list:
        training_set = MyDataset(params,dataset_dict['train'],subset='Train',aug=True)
        dataloader_dict['train'] = DataLoader(training_set, **params_train)            

    if 'validation' in subset_list:
        val_set = MyDataset(params,dataset_dict['val'],subset='Train')
        dataloader_dict['validation'] = DataLoader(val_set, **params_val)
    if 'test' in subset_list:
        test_set = MyDataset(params,dataset_dict['test'],subset='Test')
        dataloader_dict['test'] =DataLoader(test_set, **params_test)

    return dataloader_dict

def setup_heart_dataloader(params, subset_list):
    # Import de ton nouveau loader NIfTI
    from dataloaders.heart import HeartDataset

    path = params.dataset.datapath # Récupère le chemin "./data/Task02_Heart" défini dans params
    img_size = tuple(params.dataset.inshape) # [128, 128]

    # Config Workers = 0 pour Windows
    params_train = {'batch_size': params.batch, 'shuffle': True, 'num_workers': 0}
    params_val   = {'batch_size': params.batch, 'shuffle': False, 'num_workers': 0}
    # Pour le test, batch=1 pour faciliter l'affichage
    params_test  = {'batch_size': 1,            'shuffle': False, 'num_workers': 0}

    dataloader_dict = {}

    # Train & Validation (On splitte dans NiftiDataset)
    if 'train' in subset_list or 'validation' in subset_list:
        # On crée le dataset en mode 'train' (80% des données)
        ds_full_train = HeartDataset(root_dir=path, subset='train', target_size=img_size)
        
        if 'train' in subset_list:
            dataloader_dict['train'] = DataLoader(ds_full_train, **params_train)
        
        if 'validation' in subset_list:
            # Pour simplifier ici, on utilise le même jeu pour la validation
            # (Idéalement, NiftiDataset devrait gérer un split 'val' distinct)
            dataloader_dict['validation'] = DataLoader(ds_full_train, **params_val)

    # Test (20% des données)
    if 'test' in subset_list:
        ds_test = HeartDataset(root_dir=path, subset='test', target_size=img_size)
        dataloader_dict['test'] = DataLoader(ds_test, **params_test)

    return dataloader_dict