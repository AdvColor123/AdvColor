import os
import random
import numpy as np
import cv2
import torch
from torchvision import transforms
from datasets.FASDataset import FASDataset
from utils.utils import read_cfg, build_network
import models.CDCNs as CDCNs
from filters_transform import ColorFilter
from pso import PSOTransform

np.random.seed(0)
random.seed(0)
cfg = read_cfg(cfg_file='config/CDCNpp_replay.yaml')
device = 'cuda:0'
network = build_network(cfg, device=device)
test_list = 'your/test/data/list'
val_transform = transforms.Compose([
    transforms.Resize(cfg['model']['input_size']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])
valset = FASDataset(
    test_list=test_list,
    depth_map_size=cfg['model']['depth_map_size'],
    transform=val_transform,
    smoothing=cfg['train']['smoothing']
)
valloader = torch.utils.data.DataLoader(
    dataset=valset,
    batch_size=1,
    shuffle=False,
    num_workers=2
)
model = CDCNs.CDCNpp(transform=val_transform, device=device).to(device)
ckpt = torch.load('your/ckpt/path', map_location=device)
model.load_state_dict(ckpt['state_dict'], strict=False)
model = model.to(device)


def attack(image_raw, model, img_name, org_class, org_class_prob, 
           device, adv_path, label):
    image_raw = image_raw.numpy()[0].copy()
    filter = ColorFilter(device)
    save_name = 'your/image/name'
    misclassified = 0
    alpha = filter.parameter
    target_label = torch.LongTensor([1 - label.to(torch.int).item()]).to(device)
    xx = image_raw.copy().astype(np.float32)
    with torch.no_grad():
        my_pso = PSOTransform(model, xx, filter, target_label)
        my_pso.init_Population()
        fitness, flag, alpha, query, saved_images = my_pso.iterator()
        print(fitness)
        if flag:
            misclassified = 1
            print('success')
            print('orig label: %s, orig score: %s, current label: %s, alpha: %s' 
                  % (org_class.item(), org_class_prob.item(), 1-org_class.item(),
                     alpha.tolist()))
            for index, x in enumerate(saved_images):
                x_bgr = x[:, :, (2, 1, 0)]
                cv2.imwrite('{}/fake_{}_{}.png'.format(adv_path, save_name, str(index).zfill(3)), x_bgr)
        return misclassified, alpha, query


if __name__ == '__main__':
    output_path = 'your/output/path'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    misleads = 0
    spoofed = 0
    pred_labels = []
    target_labels = []
    pred_scores = []
    trials = []
    for i, (img, _, label, img_name, img_ori) in enumerate(valloader):
        print('Image: ', i)
        img_name = img_name[0]
        img, label = img.to(device), label.to(device)
        outputs = model(img_ori)
        score = torch.mean(outputs[0], axis=(1, 2))
        pred = (score >= 0.5).to(int)
        if label == 1:
            continue
        spoofed += 1
        org_class = pred
        org_class_prob = score
        mislead, alpha, query = attack(img_ori, model, img_name, org_class, org_class_prob, 
                                       device, output_path, label)
        trials.append(query)
        misleads += mislead
        print('{}\t{}\t{:.5f}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{}\t'
              .format(img_name, org_class.item(), org_class_prob.item(), 1-org_class.item(), 
                      alpha[0], alpha[1], alpha[2], query))
    avg_query = np.mean(np.array(trials))
    print('Filter: color_filter, Avg_query: %s, Mislead: %s, All: %s, Success rate: %.1f' 
          % (avg_query, misleads, spoofed, (100*float(misleads) / spoofed)) + '%')
