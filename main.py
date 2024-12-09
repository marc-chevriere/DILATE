import warnings
import warnings; warnings.simplefilter('ignore')
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt

from data.synthetic_dataset.synthetic_dataset import create_synthetic_dataset, SyntheticDataset
from models.train_eval import train_model, eval_model
from models.seq2seq import EncoderRNN, DecoderRNN, Net_GRU
from visualization.visu import plot_preds
from eval.eval_metrics import hausdorff_distance, ramp_score


def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="training script")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        metavar="D",
        help="Type of data: synthetic, traffic, other",
    )
    args = parser.parse_args()
    return args


# Visualize results
gen_test = iter(testloader)
test_inputs, test_targets, breaks = next(gen_test)

test_inputs  = torch.tensor(test_inputs, dtype=torch.float32).to(device)
test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)
criterion = torch.nn.MSELoss()

nets = [net_gru_mse,net_gru_dilate]

for ind in range(1,51):
    plt.figure()
    plt.rcParams['figure.figsize'] = (17.0,5.0)  
    k = 1
    for net in nets:
        pred = net(test_inputs).to(device)

        input = test_inputs.detach().cpu().numpy()[ind,:,:]
        target = test_targets.detach().cpu().numpy()[ind,:,:]
        preds = pred.detach().cpu().numpy()[ind,:,:]

        plt.subplot(1,3,k)
        plt.plot(range(0,N_input) ,input,label='input',linewidth=3)
        plt.plot(range(N_input-1,N_input+N_output), np.concatenate([ input[N_input-1:N_input], target ]) ,label='target',linewidth=3)   
        plt.plot(range(N_input-1,N_input+N_output),  np.concatenate([ input[N_input-1:N_input], preds ])  ,label='prediction',linewidth=3)       
        plt.xticks(range(0,40,2))
        plt.legend()
        k = k+1

    plt.show()



def main():
    args = opts()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(0)

    gamma = 0.01
    alpha = 0.5

    if args.data == "synthetic":
        # parameters
        batch_size = 100
        N = 500
        N_input = 20
        N_output = 20  
        sigma = 0.01

        # Load synthetic dataset
        X_train_input,X_train_target,X_test_input,X_test_target,train_bkp,test_bkp = create_synthetic_dataset(N,N_input,N_output,sigma)
        dataset_train = SyntheticDataset(X_train_input,X_train_target, train_bkp)
        dataset_test  = SyntheticDataset(X_test_input,X_test_target, test_bkp)
        trainloader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=1)
        validloader = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=1) 
        testloader  = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=1) 

    elif args.data == "traffic":
        pass

    else:
        pass

    if args.train == True:

        encoder_dilate = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
        decoder_dilate = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
        net_gru_dilate = Net_GRU(encoder_dilate,decoder_dilate, N_output, device).to(device)

        encoder_mse = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
        decoder_mse = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
        net_gru_mse = Net_GRU(encoder_mse,decoder_mse, N_output, device).to(device)

        train_model(
            net_gru_dilate,
            loss_type='dilate',
            learning_rate=0.001,
            trainloader=trainloader,
            validloader=validloader,
            device=device,
            epochs=500, 
            gamma=gamma, 
            alpha=alpha,
            print_every=50, 
            verbose=1,
            )
        
        train_model(
            net=net_gru_mse,
            loss_type='mse',
            learning_rate=0.001,
            trainloader=trainloader,
            validloader=validloader,
            device=device,
            epochs=500, 
            gamma=gamma, 
            alpha=alpha,
            print_every=50, 
            verbose=1,
            )
    else:
        pass

    dilate_mse, dilate_dtw, dilate_tdi = eval_model(net_gru_dilate, testloader, gamma)
    mse_mse, mse_dtw, mse_tdi = eval_model(net_gru_mse, testloader, gamma)

    ### Evaluate with Hausdorff distance and ramp score



if __name__ == "__main__":
    main()
