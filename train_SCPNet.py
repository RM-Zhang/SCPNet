import os, time
import argparse
import torch
from torch import optim
import numpy as np
import time
import random
import dataloader.datasets_homo as datasets
from network.SCPNet import SCPNet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


###### train ######
def train(args):
    
    device = torch.device('cuda:'+ str(args.gpuid[0]))

    train_loader = datasets.fetch_dataloader(args, split="train")

    net = SCPNet(args)
    net = net.to(device)
    net.train()
    print(f"Parameter Count: {count_parameters(net)}")

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps+100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    print_fre = args.print_fre
    save_fre = args.save_fre
    print("start training")

    glob_iter = 0
    train_mace12 = 0.0
    train_mace11 = 0.0
    train_mace22 = 0.0
    train_loss = 0.0
    train_unsuper_loss = 0.0
    start_time = time.time()
    
    while glob_iter <= args.num_steps:
        for i, batch_in in enumerate(train_loader):
            
            end_time = time.time()
            time_remain = (end_time - start_time) * (args.num_steps - glob_iter)
            start_time = time.time()
            
            for key, value in batch_in.items():
                batch_in[key] = batch_in[key].to(device)

            optimizer.zero_grad()
            
            batch_out = net(batch_in, 'train')

            mace12 = batch_out['mace12'].detach().cpu().mean()
            mace11 = batch_out['mace11'].detach().cpu().mean()
            mace22 = batch_out['mace22'].detach().cpu().mean()
            loss_pair12 = batch_out['loss_pair12']
            loss_pair11 = batch_out['loss_pair11']
            loss_pair22 = batch_out['loss_pair22']

            loss = 0.0 
            if args.unsupervised == 'True':
                loss += loss_pair12
            if args.supervised == 'True':
                loss += 0.1*(loss_pair11 + loss_pair22)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()
            scheduler.step()

            train_mace12 += mace12.item()
            train_mace11 += mace11.item()
            train_mace22 += mace22.item()
            train_loss += loss.item()
            train_unsuper_loss += loss_pair12.item()

            if (glob_iter + 1) % print_fre == 0:
                print(
                    "Training: Iter[{:0>3}]/[{:0>3}] "
                    "mace12: {:.4f} mace11: {:.4f} mace22: {:.4f} loss sum: {:.4f} loss unsuper: {:.4f} lr={:.8f} time: {:.2f}h".format(
                    glob_iter + 1, args.num_steps, 
                    train_mace12 / print_fre, train_mace11 / print_fre, train_mace22 / print_fre, 
                    train_loss / print_fre, train_unsuper_loss / print_fre,
                    scheduler.get_lr()[0], time_remain/3600))
                train_loss = 0.0
                train_mace12 = 0.0
                train_mace11 = 0.0
                train_mace22 = 0.0
                train_unsuper_loss = 0.0

            # save model
            if glob_iter % save_fre == 0 and glob_iter != 0:
                filename = 'models/' + 'model' + '_iter_' + str(glob_iter) + '.pth'
                model_save_path = os.path.join(args.model_dir, filename)
                checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                }
                torch.save(checkpoint, model_save_path)
                test(args, net)
                net.train()

            glob_iter += 1
            if glob_iter >= args.num_steps:
                break
    print("end training")


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
    parser.add_argument('--model_dir', type=str, default='./result/exp_ggmap', help='The models path')
    # Dataset
    parser.add_argument('--dataset', type=str, default='ggmap', help='dataset')
    parser.add_argument('--p_slf', type=int, default=32, help='homo_perturb_self')
    parser.add_argument('--p_crs', type=int, default=32, help='homo_perturb_cross')
    parser.add_argument('--pds_delta', type=int, default=32, help='illumination change')
    parser.add_argument('--batch_size', type=int, default=8)
    # Loss
    parser.add_argument('--supervised', type=str, default="True")
    parser.add_argument('--unsupervised', type=str, default="True")
    # Training
    parser.add_argument('--print_fre', type=int, default=100)
    parser.add_argument('--save_fre', type=int, default=10000)
    parser.add_argument('--num_steps', type=int, default=120000)
    parser.add_argument('--lr', type=float, default=4e-4, help='Max learning rate')
    parser.add_argument('--clip', type=float, default=1.0)
    args = parser.parse_args()
    
    setup_seed(args.seed)    
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.model_dir+'/models/'):
        os.makedirs(args.model_dir+'/models/')

    argsDict = args.__dict__
    with open(args.model_dir+'/setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')
        
    train(args)


if __name__ == "__main__":
    main()


