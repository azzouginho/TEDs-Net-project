from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List

# --- 1. La Config du Dataset ---
@dataclass_json
@dataclass
class Heart_dataset:
    ''' Paramètres spécifiques pour Task02_Heart '''
    ndims: int = 2
    inshape: List[int] = field(default_factory=lambda: [128, 128]) # Taille définie dans votre loader
    # Chemin vers le dossier dézippé
    datapath: str = "./data/Task02_Heart" 
    # Topologie : 1 composant (l'atrium), 0 trou
    betti: List[int] = field(default_factory=lambda: [1, 0])

# --- 2. L'Architecture TEDS (Adaptée) ---
@dataclass_json
@dataclass
class TEDS_Arch:
    act: int = 1 
    diffeo_int: int = 8
    guas_smooth: int = 1 
    Guas_kernel: int = 3
    sigma: float = 2.0
    mega_P: int = 2 
    dec_depth: List[int] = field(default_factory=lambda: [4, 2])

# --- 3. Réseau Général ---
@dataclass_json
@dataclass
class GeneralNet:
    dropout: int = 1
    fi: int = 12
    net_depth: int = 4 
    in_chan: int = 1 
    out_chan: int = 1 

@dataclass(frozen=True)
class LossParams:
    loss: List[str] = field(default_factory=lambda: ["dice", "grad", "grad"])
    weight: List = field(default_factory=lambda: [1,10000,10000])

# --- 4. La Classe Globale ---
@dataclass_json
@dataclass
class Parameters:
    epoch: int = 100
    lr: float = 0.0001
    lr_sch: bool = False
    batch: int = 4         # Attention à la mémoire, 4 ou 8 c'est bien
    checkpoint_freq: int = 50
    threshold: float = 0.3
    data_path: str = "./saved_models_heart" # Dossier de sauvegarde

    loss_params: LossParams = field(default_factory=LossParams)
    network_params: GeneralNet = field(default_factory=GeneralNet)

    # C'est ici qu'on force l'utilisation de Heart
    data: str = 'Heart'
    dataset: Heart_dataset = field(default_factory=Heart_dataset)
    
    net: str = 'teds'
    network: TEDS_Arch = field(default_factory=TEDS_Arch)