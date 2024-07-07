import random
import math
import numpy as np
import torch
import cv2
from models.face_normalizer import FaceNormalizer
from models.tf_face_detection_mtcnn import TfFaceDetectorMtcnn
from models.tf_face_quality_model import TfFaceQaulityModel


face_detector = TfFaceDetectorMtcnn()
face_normalizer = FaceNormalizer()
face_qaulity_predictor = TfFaceQaulityModel()


def cal_face_quality(img):
    img = img[:, :, (2, 0, 1)]
    bboxes, points = face_detector.get_face_bboxes(img, detect_scale=0.3, verbose=0)
    if len(bboxes) == 0:
        return 0
    img_align_list = face_normalizer.get_face_aligned(img, points)
    for img_align in img_align_list:
        quality_score = face_qaulity_predictor.inference(img_align)
        return quality_score
    return 0


class PSOTransform():
    def __init__(self, model, xx, filter, target_label, pN=30, dim=3,
                 max_iter=100, num_samples=40, eta=0.6):
        self.model = model
        self.xx = xx
        self.filter = filter
        self.target_label = target_label
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = 0.6
        self.r2 = 0.3
        self.pN = pN
        self.dim = dim
        self.max_iter = max_iter
        self.X = np.zeros((self.pN, self.dim))
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))
        self.gbest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)
        self.fit = 1e10
        self.flag = False
        self.query = 0
        self.num_samples = num_samples
        self.eta = eta
        self.error_rate = 1e10
        self.quality = 1e10

    def brightness_transform(self, img, brightness_factor=1.0):
        brightness_factor = random.uniform(0.2, 1.8)
        img = img * brightness_factor
        img[img > 255] = 255
        return img

    def gamma_transform(self, img, gamma=2.2):
        gamma = random.uniform(1, 3)
        img = ((img / 255)**(1 / gamma)) * 255
        img[img > 255] = 255
        return img

    def shear_transform(self, img, shear_range=5):
        rows, cols, ch = img.shape
        pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
        pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
        pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
        pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
        shear_M = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(img, shear_M, (cols, rows))

    def translation_transform(self, img, trans_range=10):
        rows, cols, ch = img.shape
        tr_x = trans_range * np.random.uniform() - trans_range / 2
        tr_y = trans_range * np.random.uniform() - trans_range / 2
        Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
        return cv2.warpAffine(img, Trans_M, (cols, rows))

    def rotate_transform(self, img, angle=0):
        angle = random.randint(-60, 60)
        rows, cols, ch = img.shape
        rotation_matrix = cv2.getRotationMatrix2D(
            (cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(img, rotation_matrix, (cols, rows))

    def blur_transform(self, img):
        k = random.randrange(3, 8, 2)
        return cv2.GaussianBlur(img, (k, k), 0)

    def light_transform(self, img):
        rows, cols, ch = img.shape
        centerX = random.randint(rows // 5, rows - rows // 5)
        centerY = random.randint(cols // 5, cols - cols // 5)
        radius = min(centerX, centerY)
        strength = random.randint(0, 300)
        dst = np.zeros((rows, cols, 3), dtype='uint8')
        for i in range(rows):
            for j in range(cols):
                distance = math.pow((centerY - j), 2) + math.pow((centerX - i), 2)
                R = img[i, j][0]
                G = img[i, j][1]
                B = img[i, j][2]
                if (distance < radius * radius):
                    result = (int)(strength * (1.0 - math.sqrt(distance) / radius))
                    R = img[i, j][0] + result
                    G = img[i, j][1] + result
                    B = img[i, j][2] + result
                    R = min(255, max(0, R))
                    G = min(255, max(0, G))
                    B = min(255, max(0, B))
                    dst[i, j] = np.uint8((R, G, B))
                else:
                    dst[i, j] = np.uint8((R, G, B))
        return dst

    def augmentation(self, img):
        r = np.random.randint(6, size=1)
        if r == 0:
            img = self.blur_transform(img)
        if r == 1:
            img = self.brightness_transform(img)
            img = self.gamma_transform(img)
        if r == 2:
            img = self.rotate_transform(img)
        if r == 3:
            img = self.translation_transform(img)
        if r == 4:
            img = self.shear_transform(img)
        if r == 5:
            img = self.light_transform(img)
        return img

    def function(self, param, save_flag=False):
        alpha = param[0:3]
        avg_loss = 0
        avg_quality = 0
        misclassified = 0
        saved_images = []
        for _ in range(self.num_samples):
            x = self.xx.copy()
            x = self.augmentation(x)
            x = x.astype(np.uint8)
            x = self.filter.filter(x, alpha)
            outputs = self.model(x)
            self.query += 1
            top1score = torch.mean(outputs[0], axis=(1, 2))
            top1label = (top1score >= 0.5).to(int)
            if top1label == self.target_label:
                misclassified += 1
                quality = 1 - \
                    cal_face_quality(x.detach().cpu().numpy().astype(np.uint8))
                quality = quality if quality > 0 else 0.0
                avg_quality += quality
                if save_flag:
                    saved_images.append(x.detach().cpu().numpy())
            logit = 1 - top1score[0]
            logit[logit < 0] = 0.0
            avg_loss += logit
        avg_loss /= self.num_samples
        error_rate_loss = 1 - misclassified / self.num_samples
        avg_quality /= (misclassified + 1e-8)
        total_loss = error_rate_loss + self.eta * avg_quality
        if save_flag:
            return total_loss, error_rate_loss, avg_quality, avg_loss, saved_images
        return total_loss, error_rate_loss, avg_quality, avg_loss

    def init_Population(self):
        self.X = np.random.uniform(0, 1, (self.pN, self.dim))
        self.V = np.random.uniform(0, 1, (self.pN, self.dim))
        for i in range(self.pN):
            self.pbest[i] = self.X[i]
            tmp, error_rate_loss, avg_quality, _ = self.function(self.X[i])
            self.p_fit[i] = tmp
            if tmp < self.fit:
                self.fit = tmp
                self.gbest = self.X[i]
                self.error_rate = error_rate_loss
                self.quality = avg_quality

    def iterator(self):
        fitness = []
        for t in range(self.max_iter):
            for i in range(self.pN):
                temp, error_rate_loss, avg_quality, _ = self.function(self.X[i])
                if temp < self.p_fit[i]:
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    if self.p_fit[i] < self.fit:
                        self.fit = self.p_fit[i]
                        self.gbest = self.X[i]
                        self.error_rate = error_rate_loss
                        self.quality = avg_quality
            self.V = self.w * self.V + self.c1 * self.r1 * \
                (self.pbest - self.X) + self.c2 * self.r2 * (self.gbest - self.X)
            self.X = self.X + self.V
            self.X[:, 0] = np.clip(self.X[:, 0], 0.0, 1.0)
            self.X[:, 1] = np.clip(self.X[:, 1], 0.0, 1.0)
            self.X[:, 2] = np.clip(self.X[:, 2], 0.0, 1.0)
            try:
                fitness.append(self.fit.item())
            except BaseException:
                fitness.append(self.fit)
            print(fitness, self.error_rate, self.quality)
            if ((abs(self.error_rate - 0.0)) < 0.1 and abs(self.quality - 0.0)
                    < 0.5) or (t > 20 and abs(fitness[-1] - fitness[-11]) < 1e-8):
                print('early converge at iter: %d, break' % t)
                if self.error_rate < 0.2:
                    self.flag = True
                break
        if fitness[-1] < 0.2:
            self.flag = True
        alpha = self.gbest[0:3]
        _, _, _, _, saved_images = self.function(alpha, save_flag=True)
        return fitness, self.flag, self.gbest, self.query, saved_images
