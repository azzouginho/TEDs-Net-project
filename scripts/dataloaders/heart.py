import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import cv2 # Pour redimensionner si besoin

class HeartDataset(Dataset):
    def __init__(self, root_dir, subset='train', target_size=(128, 128)):
        """
        Args:
            root_dir (string): Chemin vers le dossier TaskXX (ex: './data/Task02_Heart')
            subset (string): 'train' ou 'test' (on splittera manuellement)
            target_size (tuple): Taille finale de l'image 2D (H, W)
        """
        self.image_dir = os.path.join(root_dir, 'imagesTr')
        self.label_dir = os.path.join(root_dir, 'labelsTr')
        self.target_size = target_size
        
        # Récupérer la liste des fichiers
        all_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.nii.gz')])
        
        # Split simple (80% train, 20% test)
        split_idx = int(len(all_files) * 0.8)
        if subset == 'train':
            self.files = all_files[:split_idx]
        else:
            self.files = all_files[split_idx:]
            
        # Génération du Prior (Cercle centré) une seule fois
        self.prior = self._create_circle_prior(target_size)

    def _create_circle_prior(self, size):
        """Crée un disque blanc au centre d'une image noire."""
        prior = np.zeros(size, dtype=np.float32)
        center = (int(size[1]/2), int(size[0]/2)) # x, y
        radius = int(min(size) * 0.2) # Rayon = 20% de l'image
        cv2.circle(prior, center, radius, 1.0, -1) # 1.0 = Blanc, -1 = Rempli
        return prior

    def _normalize(self, img):
        """Normalisation Min-Max entre 0 et 1"""
        min_val, max_val = img.min(), img.max()
        if max_val > min_val:
            return (img - min_val) / (max_val - min_val)
        return img

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        
        # 1. Charger les volumes 3D
        img_path = os.path.join(self.image_dir, filename)
        lbl_path = os.path.join(self.label_dir, filename)
        
        # get_fdata() renvoie un tableau numpy [H, W, Profondeur]
        img_vol = nib.load(img_path).get_fdata()
        lbl_vol = nib.load(lbl_path).get_fdata()
        
        # 2. Extraire la coupe centrale (Slice)
        # On prend l'index au milieu de l'axe Z (profondeur)
        z_center = img_vol.shape[2] // 2
        img_slice = img_vol[:, :, z_center]
        lbl_slice = lbl_vol[:, :, z_center]
        
        # 3. Redimensionner (Si les images ont des tailles différentes)
        img_slice = cv2.resize(img_slice, self.target_size)
        lbl_slice = cv2.resize(lbl_slice, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # 4. Normalisation et Conversion Tensor
        img_slice = self._normalize(img_slice)
        
        # Ajout de la dimension Channel [1, H, W]
        x = torch.from_numpy(img_slice).float().unsqueeze(0)
        y = torch.from_numpy(lbl_slice).float().unsqueeze(0)
        prior = torch.from_numpy(self.prior).float().unsqueeze(0)
        
        return x, prior, y

# --- EXEMPLE D'UTILISATION ---
if __name__ == "__main__":
    # Configuration
    data_path = "./data/Task02_Heart" # Modifiez selon votre dossier
    
    dataset = HeartDataset(data_path, subset='train')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    print(f"Nombre d'images : {len(dataset)}")
    
    # Test de chargement
    x, prior, y = next(iter(loader))
    print(f"Shape Image: {x.shape}, Label: {y.shape}")
    
    # Petit affichage pour vérifier
    import matplotlib.pyplot as plt
    plt.subplot(131); plt.imshow(x[0,0], cmap='gray'); plt.title("IRM (Coeur)")
    plt.subplot(132); plt.imshow(prior[0,0], cmap='gray'); plt.title("Prior (Cercle)")
    plt.subplot(133); plt.imshow(y[0,0], cmap='gray'); plt.title("Label")
    plt.show()