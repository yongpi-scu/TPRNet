import os
import cv2
import torch
import pickle
import pydicom
import numpy as np
from PIL import Image
import SimpleITK as sitk
from torch.utils.data import Dataset
from skimage.feature import greycomatrix, greycoprops

def HuMoments(img):
    moments = cv2.moments(img)
    humoments = cv2.HuMoments(moments)  
    humoments = np.log(np.abs(humoments))
    return humoments.squeeze()

def glcm_features(img, distance=5, angle=45):
    glcm = greycomatrix(np.array(img), distances=[distance], angles=[angle], levels=256, symmetric=True, normed=False)
    contrast = greycoprops(glcm, 'contrast')[0][0]
    correlation = greycoprops(glcm, 'correlation')[0][0]
    energy = greycoprops(glcm, 'energy')[0][0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0][0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0][0]
    return [contrast,correlation,energy,dissimilarity,homogeneity]

def get_LUT_value(data, window, level):
    """Apply the RGB Look-Up Table for the given
       data and window/level value."""
    try:
        window = window[0]
    except TypeError:
        pass
    try:
        level = level[0]
    except TypeError:
        pass

    return np.piecewise(data,
                        [data <= (level - 0.5 - (window - 1) / 2),
                         data > (level - 0.5 + (window - 1) / 2)],
                        [0, 255, lambda data: ((data - (level - 0.5)) /
                                               (window - 1) + 0.5) * (255 - 0)])

class ThyroidDataset(Dataset):
    """Thyroid scintigraphy dataset."""

    def __init__(self, root_dir, pkl_file, mode="train", transforms=None):
        data_pkl = pickle.load(open(pkl_file, "rb"))
        self.data_list = data_pkl[mode]
        self.transforms = transforms
        self.root_dir = root_dir
    
    def __get_V_h(self, I_r):
        gray = np.array(I_r.convert("L").resize([100, 100]))
        # glcm 0
        glcm_0 = glcm_features(gray,angle=0)
        # glcm 45
        glcm_45 = glcm_features(gray,angle=45)
        # glcm 90
        glcm_90 = glcm_features(gray,angle=90)
        # glcm 135
        glcm_135 = glcm_features(gray,angle=135)
        # humoments
        humoments = HuMoments(gray)
        V_h = torch.tensor(np.hstack([glcm_0,glcm_45,glcm_90,glcm_135,humoments])).float()
        return V_h

    def __get_I_o(self, sample):
        dcm_img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root_dir, sample[0])))
        dcm_file = pydicom.read_file(os.path.join(self.root_dir, sample[0]))
        SmallestImagePixelValue = dcm_file.SmallestImagePixelValue
        LargestImagePixelValue = dcm_file.LargestImagePixelValue
        window = LargestImagePixelValue-SmallestImagePixelValue
        level = SmallestImagePixelValue+window//2
        I_o = Image.fromarray(get_LUT_value(dcm_img[0], window, level)).convert("RGB")
        return I_o

    def __get_I_r(self, I_o, padding_size = 5, center_crop = 0.15):
        whole_img = np.array(I_o.convert("L"))
        center_img = whole_img[int(whole_img.shape[0]*center_crop):int(whole_img.shape[0]*(1-center_crop)),
                            int(whole_img.shape[1]*center_crop):int(whole_img.shape[1]*(1-center_crop))]
        center_img = cv2.GaussianBlur(center_img, (5, 5), 0)
        otsu_threshold, image_result = cv2.threshold(center_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # print("threshold value:",otsu_threshold)
        contours, hierarchy = cv2.findContours(image_result,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        save_cnts = []
        y_max = 0
        h_max = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > center_img.size*0.002:
                x,y,w,h = cv2.boundingRect(cnt)
                y_end = y+h
                if w/h>1.5:
                    continue
                if y>y_max:
                    save_cnts=[cnt]
                    y_max = y_end
                    h_max = h
                elif abs(y_end-y_max)<max(h_max,h)*0.5:
                    save_cnts.append(cnt)
        if len(save_cnts)==1:
            x,y,w,h = cv2.boundingRect(save_cnts[0])
            left, up, right, bottom = x, y, x+w, y+h
        elif len(save_cnts)==2:
            x1,y1,w1,h1 = cv2.boundingRect(save_cnts[0])
            x2,y2,w2,h2 = cv2.boundingRect(save_cnts[1])
            left, up, right, bottom = min(x1,x2), min(y1,y2), max(x1+w1,x2+w2), max(y1+h1,y2+h2)
            if (right-left)/(bottom-up)>2:
                left, up, right, bottom = int(center_img.shape[0]*0.25),int(center_img.shape[1]*0.25),int(center_img.shape[0]*0.75),int(center_img.shape[1]*0.75)
        else:
            left, up, right, bottom = int(center_img.shape[0]*0.25),int(center_img.shape[1]*0.25),int(center_img.shape[0]*0.75),int(center_img.shape[1]*0.75)
        # pad to square
        w = right-left
        h = bottom-up
        if w>h:
            offset = int((w-h)*0.5)
            up-=offset
            bottom+=offset
        else:
            offset = int((h-w)*0.5)
            left-=offset
            right+=offset
        # aligning
        left = int(whole_img.shape[0]*center_crop)+left-padding_size
        up = int(whole_img.shape[1]*center_crop)+up-padding_size
        right = int(whole_img.shape[0]*center_crop)+right+padding_size 
        bottom = int(whole_img.shape[1]*center_crop)+bottom+padding_size
        I_r = Image.fromarray(whole_img[left:right,up:bottom]).convert("RGB")
        return I_r

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        I_o = self.__get_I_o(sample)
        I_r = self.__get_I_r(I_o)
        V_h = self.__get_V_h(I_r)
        if self.transforms is not None:
            for trans in self.transforms:
                I_o, I_r = trans([I_o, I_r])
        return I_o, I_r, V_h, sample[1]