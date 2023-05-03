import numpy as np 
from tqdm import tqdm
import cv2
import os
from matplotlib import pyplot as plt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'


TRAIN_DIR = ('D:\\Memphis\\Independent Study\\New folder (2)\\TrainSet\\X\\y\\')
TEST_DIR = ('D:\\Memphis\\Independent Study\\New folder (2)\\TestSet\\X\\y\\')

def get_amount(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray.shape
    _,thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    img_thresh_otsu = mask
    kernel = np.ones((1,50), np.uint8)
    kernel

    mask = cv2.erode(mask, kernel, iterations=1)
    mask.shape
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask
    result = cv2.bitwise_and(thresh,thresh,mask=mask)
    mask2 = cv2.bitwise_not(mask)
    result2 = cv2.bitwise_and(thresh, thresh, mask=mask2)
    img_mask = cv2.threshold(mask2,150,255, cv2.THRESH_BINARY)[1]
    diff = cv2.absdiff(img_mask,thresh)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    label_image = (255/num_labels)*labels.astype(np.uint8)
    plot = plt.imshow(label_image,cmap='gray')
    
    image = cv2.imread('output_image.png', 0)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

    # Apply thresholding to create a binary image
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Convert the binary image to RGB for compatibility with Pytesseract
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    # Use Pytesseract to extract text from the image
    text = pytesseract.image_to_string(thresh_rgb)

    # Print the extracted text
    return text