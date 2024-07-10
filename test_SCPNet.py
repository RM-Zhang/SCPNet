import os
import argparse
import torch
import numpy as np
import time
import random
import dataloader.datasets_homo as datasets
from network.SCPNet import SCPNet


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


###### test ######
def test(args, net):
    
    device = torch.device('cuda:'+ str(args.gpuid[0]))

    test_loader = datasets.fetch_dataloader(args, split="test")

    net = net.to(device)
    net.eval()

    print("start testing")
    with torch.no_grad():
        test_mace12 = 0.0
        test_mace11 = 0.0
        test_mace22 = 0.0
        for i, batch_in in enumerate(test_loader):
            
            for key, value in batch_in.items():
                batch_in[key] = batch_in[key].to(device)
            batch_out = net(batch_in, 'test')
            
            mace12 = batch_out['mace12'].detach().cpu().mean()
            mace11 = batch_out['mace11'].detach().cpu().mean()
            mace22 = batch_out['mace22'].detach().cpu().mean()
            
            test_mace12 += mace12.item()
            test_mace11 += mace11.item()
            test_mace22 += mace22.item()

    print('|Test size  |   mace12   |  mace11   |   mace22  |')
    print(len(test_loader), test_mace12 / len(test_loader), test_mace11 / len(test_loader), test_mace22 / len(test_loader))
    print("end testing")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--gpuid', type=int, nargs='+', default = [0])
    parser.add_argument('--model_dir', type=str, default='./ckpt/', help='The models path')
    parser.add_argument('--model_name', type=str, default='ggmap.pth', help='The model name')
    # Dataset
    parser.add_argument('--dataset', type=str, default='ggmap', help='dataset')
    parser.add_argument('--p_slf', type=int, default=32, help='homo_perturb_self')
    parser.add_argument('--p_crs', type=int, default=32, help='homo_perturb_cross')
    parser.add_argument('--pds_delta', type=int, default=32, help='illumination change')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    setup_seed(args.seed)    

    net = SCPNet(args)
    state_dict = torch.load(args.model_dir + args.model_name)
    net.load_state_dict(state_dict['net'])
    test(args, net)


if __name__ == "__main__":
    main()


