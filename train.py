import argparse
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from utilities.dataset import MBM
from model import ModelCountception
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch Count training')
# pickle file containing the data
parser.add_argument('--pkl-file', default="utils/MBM-dataset.pkl", type=str, help='path to pickle file.')
# batch size 
parser.add_argument('--batch-size', default=2, type=int, help='the batch size for training.')
# number of time periods (epochs)
parser.add_argument('--epochs', default=1000, type=int, help='total number of training epochs.')
# learning rate
parser.add_argument('--lr', default=0.001, type=float, help='learning rate.')


def main():
    # device to use for training the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get command line arguments
    args = parser.parse_args()
    # dataset to train on 
    training_dataset = MBM(pkl_file=args.pkl_file, transform=transforms.Compose([transforms.ToTensor()]), mode='train')
    training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    validation_dataset = MBM(pkl_file=args.pkl_file, transform=transforms.Compose([transforms.ToTensor()]), mode='valid')
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    # We use l1 regression 
    criterion = nn.L1Loss()
    model = ModelCountception().to(device)
    solver = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):

        for index, (input_, target, _) in enumerate(training_dataloader):
            input_ = input_.to(device)
            target = target.to(device)
            output = model.forward(input_)
            loss = criterion(output, target)

            # Zero grad
            model.zero_grad()
            # compute the gradients 
            loss.backward()
            # update the paramters 
            solver.step()


        with torch.no_grad():
            validation_loss = []
            for index, (input_, target, _) in enumerate(validation_dataloader):
                input_ = input_.to(device)
                target = target.to(device)
                output = model.forward(input_)
                validation_loss.append(criterion(output, target).data.cpu().numpy())

            print("Epoch", epoch, "- Validation Loss:", np.mean(validation_loss))

        if (epoch+1) % 50 == 0:
            state = {'model_weights': model.state_dict()}
            torch.save(state, "models/{0}_epochs.model".format(epoch))


if __name__ == '__main__':
    main()
