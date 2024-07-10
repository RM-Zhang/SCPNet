import cv2
import numpy as np
import torch
import torchvision
import torch.utils.data as data
import random
from glob import glob
import os.path as osp
import albumentations as A

marginal = 32
patch_size = 128

class homo_dataset(data.Dataset):
    def __init__(self):
        self.p_slf = 32
        self.p_crs = 32
        self.pds_delta = 32
        self.image_list_img1 = []
        self.image_list_img2 = []
        self.dataset=[]
        self.transform = A.Compose([A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, brightness_by_max=False, always_apply=True),
                                    A.Sharpen()])

    def __getitem__(self, index):

        img1 = cv2.imread(self.image_list_img1[index])
        img2 = cv2.imread(self.image_list_img2[index])
        if self.dataset=='ggmap':
            img1 = cv2.resize(img1, (192, 192))
            img2 = cv2.resize(img2, (192, 192))
        if self.dataset=='rgbnir':
            img1 = cv2.resize(img1, (256, 256))
            img2 = cv2.resize(img2, (256, 256))
        if self.dataset=='harvard':
            img1 = cv2.resize(img1, (348, 260))
            img2 = cv2.resize(img2, (348, 260))
        if self.dataset=='flash':
            img1 = cv2.resize(img1, (320, 213))
            img2 = cv2.resize(img2, (320, 213))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = np.expand_dims(img1, 2)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2 = np.expand_dims(img2, 2)
        
        # warp_img generated from img2, non_warp_img generated from img1
        pair12_patch_w, pair12_patch_nw, pair12_img_w, pair12_img_nw, gt12, org12, dst12 = \
            self.syn_homo(img1, img2, self.p_crs, patch_size, marginal)
        pair11_patch_w, pair11_patch_nw, pair11_img_w, pair11_img_nw, gt11, org11, dst11 = \
            self.syn_homo(img1, self.transform(image=img1)['image'], self.p_slf, patch_size, marginal)
        pair22_patch_w, pair22_patch_nw, pair22_img_w, pair22_img_nw, gt22, org22, dst22 = \
            self.syn_homo(img2, self.transform(image=img2)['image'], self.p_slf, patch_size, marginal)
                
        return {"pair12_patch_w":pair12_patch_w, "pair12_patch_nw":pair12_patch_nw, "gt12":gt12, "org12":org12, "dst12":dst12,
                "pair11_patch_w":pair11_patch_w, "pair11_patch_nw":pair11_patch_nw, "gt11":gt11, 
                "pair22_patch_w":pair22_patch_w, "pair22_patch_nw":pair22_patch_nw, "gt22":gt22, 
                }
    
    def syn_homo(self, img1, img2, perturb, patch_size, marginal):
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
        warped_img = torch.from_numpy(warped_img).float().permute(2, 0, 1)
        non_warped_img = torch.from_numpy(non_warped_img).float().permute(2, 0, 1)

        return warped_patch, non_warped_patch, warped_img, non_warped_img, ground_truth, org_pts, dst_pts


class MYDATA(homo_dataset):
    def __init__(self, split='train', dataset='ggmap', p_slf=32, p_crs=32, pds_delta=32):
        super(MYDATA, self).__init__()
        if split == 'train':   
            if dataset=='ggmap':
                root_img1 = '/data/data0/zrm/homography/datasets/ggmap/trainA'
                root_img2 = '/data/data0/zrm/homography/datasets/ggmap/trainB'
            if dataset=='rgbnir':
                root_img1 = '/data/data0/zrm/homography/datasets/RGBNIR/trainA'
                root_img2 = '/data/data0/zrm/homography/datasets/RGBNIR/trainB'
            if dataset=='harvard':
                root_img1 = '/data/data0/zrm/homography/datasets/Harvard/train/p1'
                root_img2 = '/data/data0/zrm/homography/datasets/Harvard/train/p2'
            if dataset=='flash':
                root_img1 = '/data/data0/zrm/homography/datasets/Flash_no_flash/trainA'
                root_img2 = '/data/data0/zrm/homography/datasets/Flash_no_flash/trainB' 
        else:
            if dataset=='ggmap':
                root_img1 = '/data/data0/zrm/homography/datasets/ggmap/testA'
                root_img2 = '/data/data0/zrm/homography/datasets/ggmap/testB'
            if dataset=='rgbnir':
                root_img1 = '/data/data0/zrm/homography/datasets/RGBNIR/testA'
                root_img2 = '/data/data0/zrm/homography/datasets/RGBNIR/testB'
            if dataset=='harvard':
                root_img1 = '/data/data0/zrm/homography/datasets/Harvard/test/p1'
                root_img2 = '/data/data0/zrm/homography/datasets/Harvard/test/p2'
            if dataset=='flash':
                root_img1 = '/data/data0/zrm/homography/datasets/Flash_no_flash/testA'
                root_img2 = '/data/data0/zrm/homography/datasets/Flash_no_flash/testB' 
            
        self.dataset = dataset
        self.p_slf = p_slf
        self.p_crs = p_crs
        self.pds_delta = pds_delta
        
        if dataset=='rgbnir':
            self.image_list_img1 = sorted(glob(osp.join(root_img1, '*.bmp')))
            self.image_list_img2 = sorted(glob(osp.join(root_img2, '*.bmp')))
        elif dataset=='harvard':
            self.image_list_img1 = sorted(glob(osp.join(root_img1, '*.png')))
            self.image_list_img2 = sorted(glob(osp.join(root_img2, '*.png')))
        else:
            self.image_list_img1 = sorted(glob(osp.join(root_img1, '*.jpg')))
            self.image_list_img2 = sorted(glob(osp.join(root_img2, '*.jpg')))
    
    def __len__(self):
        return int(len(self.image_list_img1))


def fetch_dataloader(args, split='train'):
    if split == 'train':
        train_dataset = MYDATA(split='train', dataset=args.dataset, p_slf=args.p_slf, p_crs=args.p_crs, pds_delta=args.pds_delta)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                       pin_memory=True, shuffle=True, num_workers=8, drop_last=False)
        print('Training with %d image pairs' % len(train_dataset))
    else: 
        train_dataset = MYDATA(split='val', dataset=args.dataset, p_slf=args.p_slf, p_crs=args.p_crs, pds_delta=args.pds_delta)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                       pin_memory=True, shuffle=True, num_workers=8, drop_last=False)       
    return train_loader


##### pdscoco #####
class ImageConvertFromInts(object):
    def __call__(self, image):
        return image.astype(np.float32)


class ImageCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ImageRandomBrightness(object):
    def __init__(self, max_delta=32, random_state=None):
        assert max_delta >= 0.0
        assert max_delta <= 255.0
        self.delta = max_delta
        self.random_state = random_state

    def __call__(self, image):
        if self.random_state.randint(2):
            delta = self.random_state.uniform(-self.delta, self.delta)
            image += delta
        return image


class ImageRandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5, random_state=None):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
        self.random_state = random_state

    # expects float image
    def __call__(self, image):
        if self.random_state.randint(2):
            alpha = self.random_state.uniform(self.lower, self.upper)
            image *= alpha
        return image


class ImageConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image


class ImageRandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5, random_state=None):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
        self.random_state = random_state

    def __call__(self, image):
        if self.random_state.randint(2):
            image[:, :, 1] *= self.random_state.uniform(self.lower, self.upper)
        return image


class ImageRandomHue(object):
    def __init__(self, delta=18.0, random_state=None):
        assert 0.0 <= delta <= 360.0
        self.delta = delta
        self.random_state = random_state

    def __call__(self, image):
        if self.random_state.randint(2):
            image[:, :, 0] += self.random_state.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image


class ImageSwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        image = image[:, :, self.swaps]
        return image


class ImageRandomLightingNoise(object):
    def __init__(self, random_state):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
        self.random_state = random_state

    def __call__(self, image):
        if self.random_state.randint(2):
            swap = self.perms[self.random_state.randint(len(self.perms))]
            shuffle = ImageSwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image


class PhotometricDistort(object):
    def __init__(self, max_delta):
        self.max_delta = max_delta

        lower = 1.0 - self.max_delta / 32 * 0.5
        upper = 1.0 + self.max_delta / 32 * 0.5

        self.random_state = np.random.RandomState(0)
        self.pd = [
            ImageRandomContrast(lower=lower, upper=upper, random_state=self.random_state),  # RGB
            ImageConvertColor(current="RGB", transform='HSV'),  # HSV
            ImageRandomSaturation(lower=lower, upper=upper, random_state=self.random_state),  # HSV
            ImageRandomHue(delta=max_delta/2, random_state=self.random_state),  # HSV
            ImageConvertColor(current='HSV', transform='RGB'),  # RGB
            ImageRandomContrast(lower=lower, upper=upper, random_state=self.random_state)  # RGB
        ]
        self.from_int = ImageConvertFromInts()
        self.rand_brightness = ImageRandomBrightness(max_delta=max_delta, random_state=self.random_state)
        self.rand_light_noise = ImageRandomLightingNoise(random_state=self.random_state)

    def __call__(self, im):
        im = im.copy()
        im = self.from_int(im)
        im = self.rand_brightness(im)
        if self.random_state.randint(2):
            distort = ImageCompose(self.pd[:-1])
        else:
            distort = ImageCompose(self.pd[1:])
        im = distort(im)
        im = self.rand_light_noise(im)
        return im