import easyocr
import cv2
from PIL import Image
import numpy as np
from langdetect import detect
from spellchecker import SpellChecker
import os
from Levenshtein import distance as levenshtein_distance

# Initialize OCR reader
languages = ['en']
reader = easyocr.Reader(languages)

# File upload (use a local path instead of google.colab's file upload)
image_path = 'path_to_your_image.jpg'  # Update this with your image file path
print("Image Path:", image_path)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9 , 75, 75)
    alpha = 1.5
    beta = 10
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary

processed_image = preprocess_image(image_path)

def ocr_image(image_path):
    processed_image = preprocess_image(image_path)
    results = reader.readtext(processed_image)
    extracted_text = " ".join([result[1] for result in results])
    return extracted_text, results

extracted_text, ocr_results = ocr_image(image_path)
print("Extracted Text:", extracted_text)

def detect_language(text):
    try:
        return detect(text)
    except:
        return "Unknown"

detected_language = detect_language(extracted_text)
print("Detected Language:", detected_language)

# Visualize results (using OpenCV to display image locally)
def visualize_results(image_path, results):
    image = cv2.imread(image_path)
    for (bbox, text, prob) in results:
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Display image using OpenCV (this will open a window with the image)
    cv2.imshow("OCR Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

visualize_results(image_path, ocr_results)

# Example ground truth text
ground_truth = "你被关在一个小房间里。你并不记得发生了什么,也不知道为什么被关在这里。你以前从房门的窗口那儿得到食物,但是你用力敲门或者大叫都没有用。你决定一定要逃跑,要不然情况可能会变更不好。"

# Calculate accuracy using Levenshtein Distance
def calculate_accuracy(ground_truth, extracted_text):
    edit_distance = levenshtein_distance(ground_truth, extracted_text)
    char_accuracy = (1 - edit_distance / max(len(ground_truth), len(extracted_text))) * 100
    return char_accuracy

char_accuracy = calculate_accuracy(ground_truth, extracted_text)
print("Character-Level Accuracy:", char_accuracy, "%")
