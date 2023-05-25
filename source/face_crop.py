from mtcnn import MTCNN
import numpy as np
import os
import cv2
import glob
import shutil
detector = MTCNN()
data = glob.glob("./dataset_train/no_mask/*")
for a_img in data:
    # print(a_img)
    img_read = cv2.imread(a_img)
    img_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
    bboxes = detector.detect_faces(img_read)
    try:
        if len(bboxes) != 0:
            for bboxe in bboxes:
                bbox = bboxe['box']
                bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])  
                img_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)         
                img2 = img_read[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                # img2 = cv2.resize(img2, (160, 160))
                cv2.imwrite(a_img, img2)
        else:
            continue
    except:
        print("aaaa")
