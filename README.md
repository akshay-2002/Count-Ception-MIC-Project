Paper implemented:https://arxiv.org/pdf/1703.08710.pdf

## Training

To train a model, run the following command: 

`python train.py --pkl-file 'utilities/MBM-dataset.pkl' --batch-size 2 --epochs 1000 --lr 0.001`

To test the model, run the following command:

`python test.py --pkl-file 'utilities/MBM-dataset.pkl' --batch-size 1 --ckpt 'models/950_epochs.model'`

