import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from network.TEDS_Net import TEDS_Net
from dataloaders.setup import setup_mnist_dataloader 
from parameters.mnist_parameters import Parameters, MNIST_dataset, TEDS_Arch, GeneralNet

def configure_mnist_params():
    """
    Configuration pour TEDS-Net sur MNIST (Standard / Base).
    """
    print("⚙️ Configuration des paramètres MNIST Standard...")
    params = Parameters()
    
    # 1. Configurer pour MNIST
    params.data = 'mnist'  # Minuscule souvent préférée par les scripts
    params.batch = 1
    
    # Dataset MNIST (28x28, line_thick=3)
    params.dataset = MNIST_dataset()
    
    # 2. Architecture STANDARD (Modèle de base)
    params.network = TEDS_Arch()
    
    # --- PARAMÈTRES CLÉS DU PAPIER POUR MNIST ---
    params.network.sigma = 2.0       # Sigma positif = Lissage Gaussien Fixe
    params.network.Guas_kernel = 3   # Noyau plus petit pour du 28x28
    params.network.dec_depth = [1]   # MNIST est simple, souvent 1 seule branche
    params.network.diffeo_int = 8    # Valeur par défaut pour MNIST
    params.network.mega_P = 2
    
    # Réseau général (Plus léger pour MNIST)
    params.network_params = GeneralNet()
    params.network_params.fi = 12        # 12 Filtres (contre 32 pour ACDC)
    params.network_params.net_depth = 2  # Profondeur 2 (contre 4 pour ACDC)
    
    return params

def save_prediction_plot(x, prior, label, pred, index, save_dir):
    """Sauvegarde une image comparative."""
    img = x[0, 0].cpu().numpy()
    pr = prior[0, 0].cpu().numpy()
    lbl = label[0, 0].cpu().numpy()
    
    # Binarisation
    res = (pred[0, 0].cpu().numpy() > 0.5).astype(float)

    fig, ax = plt.subplots(1, 4, figsize=(12, 4)) # Plus petit car images 28x28
    
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title(f"Chiffre #{index}")
    
    ax[1].imshow(pr, cmap='gray')
    ax[1].set_title("Prior")
    
    # Superposition
    rgb = np.zeros((lbl.shape[0], lbl.shape[1], 3))
    rgb[..., 1] = lbl  # Vert
    rgb[..., 0] = res  # Rouge
    
    ax[2].imshow(rgb)
    ax[2].set_title("Pred (R) vs GT (V)")
    
    # Erreurs
    err = np.abs(lbl - res)
    ax[3].imshow(err, cmap='Reds')
    ax[3].set_title("Erreurs")
    
    for a in ax: a.axis('off')
    
    filename = os.path.join(save_dir, f"mnist_res_{index:03d}.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

def run_evaluation():
    # 1. Init
    params = configure_mnist_params()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = "./test_results_mnist"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Création du modèle (Taille adaptée à MNIST)
    model = TEDS_Net(params).to(device)
    
    # 3. Chargement des poids
    model_path = 'best_model.pth'
    
    if os.path.exists(model_path):
        print(f"✅ Chargement du modèle : {model_path}")
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"❌ Erreur de dimension ! Vérifie que tu as bien entraîné avec ces paramètres (fi=12, depth=2).")
            print(f"Détail : {e}")
            return
    else:
        print("⚠️  Pas de fichier .pth trouvé. Test avec poids aléatoires.")

    model.eval()

    # 4. Chargement des données MNIST
    print("⏳ Chargement du Dataset MNIST...")
    # Le setup_mnist_dataloader retourne souvent directement le dico ou un tuple, vérifie ton setup.py
    # Ici je suppose qu'il renvoie un dictionnaire comme pour ACDC
    loaders = setup_mnist_dataloader(params, ['test'])
    test_loader = loaders['test']
    test_loader.num_workers = 0
    test_loader.shuffle = True
    
    # 5. Inférence
    print(f"🚀 Test sur MNIST (Sauvegarde dans {output_dir})")
    print(test_loader)
    count = 0
    max_images = 10
    
    with torch.no_grad():
        for (x, prior_shape, labels) in tqdm(test_loader, total=max_images):
            if count >= max_images: break
            
            x, prior_shape = x.to(device), prior_shape.to(device)
            
            outputs = model(x, prior_shape)
            pred = outputs[0] if isinstance(outputs, tuple) else outputs
            
            save_prediction_plot(x, prior_shape, labels, pred, count, output_dir)
            count += 1

    print(f"\n✅ Fini ! Résultats dans '{output_dir}'.")

run_evaluation()