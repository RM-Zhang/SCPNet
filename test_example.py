import os
import argparse
import torch
import numpy as np
import time
import random
import cv2
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


def syn_homo(img1, img2, perturb, patch_size, marginal):
    # [-1, 1]
    img1, img2 = 2*(img1/255.0)-1, 2*(img2/255.0)-1
    
    (height, width, _) = img1.shape
    x = random.randint(marginal, width - marginal - patch_size)
    y = random.randint(marginal, height - marginal - patch_size)
    top_left = (x, y)
    bottom_left = (x, patch_size + y - 1)
    bottom_right = (patch_size + x - 1, patch_size + y - 1)
    top_right = (patch_size + x - 1, y)
    four_pts = np.array([top_left, top_right, bottom_left, bottom_right])
    # crop image 192*192
    img1 = img1[top_left[1]-marginal:bottom_right[1]+marginal+1, top_left[0]-marginal:bottom_right[0]+marginal+1, :]
    img2 = img2[top_left[1]-marginal:bottom_right[1]+marginal+1, top_left[0]-marginal:bottom_right[0]+marginal+1, :]
    four_pts = four_pts - four_pts[np.newaxis, 0] + marginal # top_left -> (marginal, marginal)
    (top_left, top_right, bottom_left, bottom_right) = four_pts
    
    try:
        four_pts_perturb = []
        for i in range(4):
            t1 = random.randint(-perturb, perturb)
            t2 = random.randint(-perturb, perturb)
            four_pts_perturb.append([four_pts[i][0] + t1, four_pts[i][1] + t2])
        org_pts = np.array(four_pts, dtype=np.float32)
        dst_pts = np.array(four_pts_perturb, dtype=np.float32)
        ground_truth = dst_pts - org_pts
        H = cv2.getPerspectiveTransform(org_pts, dst_pts)
        H_inverse = np.linalg.inv(H)
    except:
        four_pts_perturb = []
        for i in range(4):
            t1 =   perturb // (i + 1)
            t2 = - perturb // (i + 1)
            four_pts_perturb.append([four_pts[i][0] + t1, four_pts[i][1] + t2])
        org_pts = np.array(four_pts, dtype=np.float32)
        dst_pts = np.array(four_pts_perturb, dtype=np.float32)
        ground_truth = dst_pts - org_pts
        H = cv2.getPerspectiveTransform(org_pts, dst_pts)
        H_inverse = np.linalg.inv(H)
    
    warped_img = cv2.warpPerspective(img2, H_inverse, (img1.shape[1], img1.shape[0]))
    warped_img = np.expand_dims(warped_img, 2)
    warped_patch = warped_img[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1, :]
    non_warped_img = img1
    non_warped_patch = non_warped_img[top_left[1]:bottom_right[1]+1, top_left[0]:bottom_right[0]+1, :]
    
    warped_patch = torch.from_numpy(warped_patch).float().permute(2, 0, 1)
    non_warped_patch = torch.from_numpy(non_warped_patch).float().permute(2, 0, 1)
    
    return warped_patch, non_warped_patch, warped_img, non_warped_img, ground_truth, org_pts, dst_pts


###### test ######
def test(args, net):
    device = torch.device('cuda:'+ str(args.gpuid[0]))

    net = net.to(device)
    net.eval()
    
    img1 = cv2.imread('./example/ggmap-modalA.jpg')
    img2 = cv2.imread('./example/ggmap-modalB.jpg')
    if args.dataset=='ggmap':
        img1 = cv2.resize(img1, (192, 192))
        img2 = cv2.resize(img2, (192, 192))
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = np.expand_dims(img1, 2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = np.expand_dims(img2, 2)

    pair12_patch_w, pair12_patch_nw, pair12_img_w, pair12_img_nw, gt12, org12, dst12 = syn_homo(img1, img2, args.p_crs, 128, 32)

    with torch.no_grad():
        _, _, pred_4p = net.network_forward(net.img_projector, net.homo_predictor, pair12_patch_w.unsqueeze(0).to(device), pair12_patch_nw.unsqueeze(0).to(device))
        pred_4p = pred_4p.squeeze(0).cpu().numpy()
        dst_pred = pred_4p + org12
        H = cv2.getPerspectiveTransform(org12, dst_pred)
        pair12_img_w_pred = cv2.warpPerspective(pair12_img_w, H, (pair12_img_w.shape[1], pair12_img_w.shape[0]))

        cv2.imwrite('./example/pair12_img_w_pred.jpg', (pair12_img_w_pred+1)/2*255)
        cv2.imwrite('./example/pair12_img_w.jpg', (pair12_img_w+1)/2*255)
        cv2.imwrite('./example/pair12_img_nw.jpg', (pair12_img_nw+1)/2*255)


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


