## TEDS-Net Implementation ##
Overview of architecture and implementation of TEDS-Net, as described in MICCAI 2021: "TEDS-Net: Enforcing Diffeomorphisms in Spatial Transformers to Guarantee TopologyPreservation in Segmentations"


Updated code (Jan 2023) now including a brief training script with a mock MNIST dataset, to perform "0" segmentation. A parameter file is also included to describe the hyper-paramters used for the ACDC training and a code for the prior shape is in the dataloader. If using the ACDC example, ensure to ammend the datapaths in both the hyperparameter file and dataloader.

Running:

>> train_runner.py

will train TEDS-Net for 20 epochs.

Running:

>> visualisation.py

will evaluate on test set and display results.
