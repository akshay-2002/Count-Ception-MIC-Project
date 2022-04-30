import argparse
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utilities.dataset import MBM
from model import ModelCountception
from utilities.save_plot import save_plot
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Sealion count training')
# Pickle data file
parser.add_argument('--pkl-file', default="utils/MBM-dataset.pkl", type=str, help='path to pickle file.')
# batch size
parser.add_argument('--batch-size', default=1, type=int, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
# checkpoint trained models
parser.add_argument('--checkpoint', default='models/950_epochs.model', type=str, help='Path to the .model file')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Get the arguments
    args = parser.parse_args()

    # Get the testing dataset from the pickle file and given batch_size
    test_dataset = MBM(pkl_file=args.pkl_file, transform=transforms.Compose([transforms.ToTensor()]), mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    # Use l1 regression
    criterion = nn.L1Loss()
    # Get the model
    model = ModelCountception().to(device)

    # Deactivate the dropout layers durin testing
    model.eval()

    # Load the model file
    from_before = torch.load(args.ckpt, map_location=torch.device('cpu' if not torch.cuda.is_available() else 'cuda'))

    # Set the model weights using above file
    model_weights = from_before['model_weights']
    model.load_state_dict(model_weights)

    # For calculating mean absolute error and total count error
    test_loss = []
    count_loss = []
    with torch.no_grad():
        for index, (input_, target, target_count) in enumerate(test_dataloader):
            input_ = input_.to(device)
            target = target.to(device)
            output = model.forward(input_)
            test_loss.append(criterion(output, target).data.cpu().numpy())
            # Size of the receptive field
            patch_size = 32

            # Redundant counts
            redundant_counts = ((patch_size / 1) ** 2.0)
            # True count
            output_count = (output.cpu().numpy() / redundant_counts).sum(axis=(2, 3))
            target_count = target_count.data.cpu().numpy()
            count_loss.append(abs(output_count - target_count))

            # save the count map plot
            save_plot(output, target, index)
        print('MAE of Test Set: ', np.mean(test_loss))
        print('Mean Difference in Counts', np.mean(count_loss))


if __name__ == '__main__':
    main()
