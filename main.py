import warnings
import warnings; warnings.simplefilter('ignore')
import argparse

import torch
from torch.utils.data import DataLoader
import random

from data.synthetic_dataset.synthetic_dataset import get_synthetic_data
from data.traffic.traffic_dataset import get_traffic_data
from models.train_eval import train_model, eval_model
from models.seq2seq import EncoderRNN, DecoderRNN, Net_GRU
from visualization.visu import plot_preds


def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="training script")
    parser.add_argument(
        "--data",
        type=str,
        default="traffic",
        metavar="D",
        help="Type of data: synthetic, traffic, other",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=3,
        metavar="D",
        help="Type of data: synthetic, traffic, other",
    )
    parser.add_argument(
        "--viz",
        type=bool,
        default=False,
        metavar="V",
        help="Visualization of predictions",
    )
    args = parser.parse_args()
    return args


def main():
    args = opts()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(0)

    n_epochs = args.n_epochs

    gamma = 0.01
    alpha = 0.5

    if args.data == "synthetic":
        output_length = 20
        trainloader, validloader, testloader = get_synthetic_data(output_length)

    elif args.data == "traffic":
        output_length = 24
        path_data = "data/traffic/traffic.txt.gz"
        trainloader, validloader, testloader = get_traffic_data(path_data=path_data, batch_size=64)

    else:
        pass


    encoder_dilate = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
    decoder_dilate = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
    net_gru_dilate = Net_GRU(encoder_dilate,decoder_dilate, output_length, device).to(device)

    encoder_mse = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
    decoder_mse = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
    net_gru_mse = Net_GRU(encoder_mse,decoder_mse, output_length, device).to(device)

    encoder_dtw = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
    decoder_dtw = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
    net_gru_dtw = Net_GRU(encoder_dtw,decoder_dtw, output_length, device).to(device)

    if args.train == True:

        train_model(
            net_gru_dilate,
            loss_type='dilate',
            learning_rate=0.001,
            trainloader=trainloader,
            validloader=validloader,
            device=device,
            epochs=n_epochs, 
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
            epochs=n_epochs, 
            gamma=gamma, 
            alpha=alpha,
            print_every=50, 
            verbose=1,
            )
        train_model(
            net=net_gru_dtw,
            loss_type='dilate',
            learning_rate=0.001,
            trainloader=trainloader,
            validloader=validloader,
            device=device,
            epochs=n_epochs, 
            gamma=gamma, 
            alpha=1,
            print_every=50, 
            verbose=1,
            )
        
        torch.save(net_gru_dilate.state_dict(), 'weights_models/net_gru_dilate.pth')
        torch.save(net_gru_mse.state_dict(), 'weights_models/net_gru_mse.pth')
        torch.save(net_gru_dtw.state_dict(), 'weights_models/net_gru_dtw.pth')

    else:
        # Chargement des poids sauvegardés
        net_gru_dilate.load_state_dict(torch.load('weights_models/net_gru_dilate.pth'))
        net_gru_mse.load_state_dict(torch.load('weights_models/net_gru_mse.pth'))
        net_gru_dtw.load_state_dict(torch.load('weights_models/net_gru_dtw.pth'))

        # Assurez-vous que les modèles sont en mode évaluation
        net_gru_dilate.eval()
        net_gru_mse.eval()
        net_gru_dtw.eval()

    dilate_mse, dilate_dtw, dilate_tdi, dilate_hausdorff, dilate_ramp= eval_model(net_gru_dilate, testloader, gamma)
    mse_mse, mse_dtw, mse_tdi, mse_hausdorff, mse_ramp = eval_model(net_gru_mse, testloader, gamma)
    dtw_mse, dtw_dtw, dtw_tdi, dtw_hausdorff, dtw_ramp = eval_model(net_gru_dtw, testloader, gamma)
    
    print("-"*50)
    print("EVALUATION")
    print("-"*50)
    print("Eval dilate")
    print('mse= ', dilate_mse ,
        ' dtw= ', dilate_dtw ,
        ' tdi= ', dilate_tdi,
        ' hausdorff= ', dilate_hausdorff ,
        ' ramp= ', dilate_ramp) 
    print("-"*50)
    print("Eval mse")
    print('mse= ', mse_mse ,
        ' dtw= ', mse_dtw ,
        ' tdi= ', mse_tdi,
        ' hausdorff= ', mse_hausdorff ,
        ' ramp= ', mse_ramp) 
    print("-"*50)
    print("Eval softDTW")
    print('mse= ', dtw_mse ,
        ' dtw= ', dtw_dtw ,
        ' tdi= ', dtw_tdi,
        ' hausdorff= ', dtw_hausdorff ,
        ' ramp= ', dtw_ramp) 
    print("-"*50)


    if args.viz:
        gen_test = iter(testloader)
        batches_to_process = 2 

        for _ in range(batches_to_process):
            test_inputs, test_targets = next(gen_test)

            test_inputs = test_inputs.to(torch.float32)
            test_targets = test_targets.to(torch.float32)

            batch_size = test_inputs.size(0)
            random_indices = torch.randint(0, batch_size, (3,)) 

            preds = {"MSE": [], "DILATE": [], "sDTW": []}

            with torch.no_grad():
                preds_mse_batch = net_gru_mse(test_inputs).squeeze(-1).detach().numpy()
                preds_dilate_batch = net_gru_dilate(test_inputs).squeeze(-1).detach().numpy()
                preds_dtw_batch = net_gru_dtw(test_inputs).squeeze(-1).detach().numpy()

                for ind in random_indices:
                    preds["MSE"].append(preds_mse_batch[ind])
                    preds["DILATE"].append(preds_dilate_batch[ind])
                    preds["sDTW"].append(preds_dtw_batch[ind])
                    X_true = test_inputs[ind].squeeze(-1).numpy()
                    y_true = test_targets[ind].squeeze(-1).numpy()
                    plot_preds(
                        X_true, 
                        y_true, 
                        {"MSE": preds["MSE"][-1], "DILATE": preds["DILATE"][-1], "sDTW": preds["sDTW"][-1]},
                        save_path="figures", 
                        file_name=f"time_series_plot_{ind}.png"
                    )


if __name__ == "__main__":
    main()
