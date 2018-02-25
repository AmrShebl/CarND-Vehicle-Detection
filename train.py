import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from scipy.ndimage.measurements import label

def get_train_and_test_data(car_folder, non_car_folder):
    car_images = []
    non_car_images = []
    for root, dirs, files in os.walk(car_folder):
        for file in files:
            if file.endswith('.png'):
                img = cv2.imread(os.path.join(root, file))
                car_images.append(img)
    for root, dirs, files in os.walk(non_car_folder):
        for file in files:
            if file.endswith('.png'):
                img = cv2.imread(os.path.join(root, file))
                non_car_images.append(img)
    X = car_images + non_car_images
    Y = [1] * len(car_images) + [0] * len(non_car_images)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
    return X_train, X_test, Y_train, Y_test

def get_image_histogram(img, n_bins=32,bins_range=(0,256)):
    ch0 = img[:,:,0]
    ch1 = img[:,:,1]
    ch2 = img[:,:,2]
    hist0 = np.histogram(ch0,n_bins,bins_range)[0]
    hist1 = np.histogram(ch1,n_bins,bins_range)[0]
    hist2 = np.histogram(ch2,n_bins,bins_range)[0]
    feat = np.concatenate((hist0,hist1, hist2))
    return feat

def get_hog_features(img, n_orientations=9,pixels_per_cell=8, cells_per_block=2, visualize=False, feature_vector=True):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if visualize:
        features, hog_img = hog(gray_img, n_orientations, (pixels_per_cell, pixels_per_cell), (cells_per_block, cells_per_block),
            visualise=visualize,feature_vector=feature_vector)
        plt.figure()
        plt.subplot('121')
        plt.imshow(img)
        plt.title('Original Image')
        plt.subplot('122')
        plt.imshow(hog_img)
        plt.title('Hog Image')
        plt.show()
    else:
        features = hog(gray_img, n_orientations, (pixels_per_cell, pixels_per_cell),
                       (cells_per_block, cells_per_block),
                       visualise=visualize, feature_vector=feature_vector)
    return features

def get_non_hog_features(img):
    my_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    bin_img = cv2.resize(my_img, (16, 16))
    feat0 = np.ravel(bin_img)
    feat1 = get_image_histogram(my_img)
    return feat0, feat1

def extract_features(img):
    feat0, feat1 = get_non_hog_features(img)
    feat2 = get_hog_features(img)
    feat = np.concatenate((feat0,feat1,feat2))
    # plt.figure()
    # x=range(len(feat))
    # plt.plot(x,feat)
    return feat

def train_classifier(car_folder, non_car_folder):
    X_train, X_test, Y_train, Y_test = get_train_and_test_data(car_folder,non_car_folder)
    X_features = []
    for img in X_train:
        feat = extract_features(img)
        X_features.append(feat)
    X_train_features = np.vstack(X_features)
    X_features = []
    for img in X_test:
        feat = extract_features(img)
        X_features.append(feat)
    X_test_features = np.vstack(X_features)
    scaler = StandardScaler().fit(X_features)
    scaled_X_train_features = scaler.transform(X_train_features)
    scaled_X_test_features = scaler.transform(X_test_features)
    clf = LinearSVC()
    clf.fit(scaled_X_train_features,Y_train)
    pred = clf.predict(scaled_X_test_features)
    accuracy = accuracy_score(Y_test, pred)
    print("The accuracy of the classifier is {}".format(accuracy))
    return clf, scaler

def draw_boxes(img, boxes, color = (0,0,255), thickness = 6):
    for box in boxes:
        cv2.rectangle(img, box[0], box[1], color, thickness)

def find_cars_with_scale(img, scale, y_start, y_end, clf, scaler):
    out_boxes=[]
    window_pixels = 64 #This is the size of the window in pixels
    #The classifier was trained on 64*64 images
    hog_window = 7
    pixel_per_cell = 8
    y_pos = 0
    step = 1 #This is the step in cells
    draw_img = np.copy(img)
    sub_img = draw_img[y_start:y_end,:,:]
    scale_sub_img = sub_img
    if scale!=1:
        scale_sub_img = cv2.resize(sub_img,(int(sub_img.shape[1]/scale), int(sub_img.shape[0]/scale)))
    # plt.figure()
    # plt.imshow(scale_sub_img)
    hog_features = get_hog_features(scale_sub_img, feature_vector=False)
    yMax, xMax, c = scale_sub_img.shape
    while y_pos+window_pixels<yMax:
        x_pos = 0
        while x_pos+window_pixels<xMax:
            window_img = scale_sub_img[y_pos:y_pos+window_pixels, x_pos:x_pos+window_pixels, :]
            cell_x_pos = x_pos//pixel_per_cell
            cell_y_pos = y_pos//pixel_per_cell
            feat0, feat1 = get_non_hog_features(window_img)
            window_hog = hog_features[cell_y_pos:cell_y_pos+hog_window,cell_x_pos:cell_x_pos+hog_window]
            features = [np.hstack((feat0, feat1, window_hog.ravel()))]
            features = scaler.transform(features)
            pred = clf.predict(features)
            box = ((int(x_pos * scale), int((y_pos * scale)+y_start)), (int((x_pos + window_pixels) * scale), int(((y_pos + window_pixels) * scale)+y_start)))
            #cv2.rectangle(draw_img,box[0], box[1],(255,0,0),6)
            if pred:
                out_boxes.append(box)

            x_pos+=step*pixel_per_cell
        y_pos+=step*pixel_per_cell
    draw_boxes(draw_img,out_boxes)
    # plt.figure()
    # draw_img = cv2.cvtColor(draw_img,cv2.COLOR_BGR2RGB)
    # plt.imshow(draw_img)
    return out_boxes

def update_heat_map(heat_map,boxes):
    for box in boxes:
        heat_map[box[0][1]:box[1][1],box[0][0]:box[1][0]]+=1

def apply_threshold(heat_map, threshold):
    heat_map[heat_map<=threshold]=0

def get_car_boxes(labels):
    map = labels[0]
    n_cars = labels[1]
    out_boxes=[]
    for car in range(1,n_cars+1):
        car_indices = np.where(map==car)
        yMin = np.min(car_indices[0])
        yMax = np.max(car_indices[0])
        xMin = np.min(car_indices[1])
        xMax = np.max(car_indices[1])
        out_boxes.append(((xMin,yMin),(xMax,yMax)))
    return out_boxes

def find_cars(img, clf, scaler):
    draw_img = np.copy(img)
    heat_map = np.zeros(img.shape[:-1])
    boxes = find_cars_with_scale(img,scale=1,y_start=400, y_end=656,clf=clf,scaler=scaler)
    update_heat_map(heat_map, boxes)
    boxes = find_cars_with_scale(img, scale=1.5, y_start=400, y_end=656, clf=clf, scaler=scaler)
    update_heat_map(heat_map, boxes)
    boxes = find_cars_with_scale(img, scale=0.75, y_start=400, y_end=656, clf=clf, scaler=scaler)
    update_heat_map(heat_map, boxes)
    apply_threshold(heat_map,20)
    labels = label(heat_map)
    car_boxes = get_car_boxes(labels)
    draw_boxes(draw_img,car_boxes,(255,0,0))
    # plt.figure()
    # plt.imshow(heat_map,cmap='hot')
    # plt.figure()
    # plt.imshow(labels[0],cmap='gray')
    # plt.figure()
    # draw_img = cv2.cvtColor(draw_img,cv2.COLOR_BGR2RGB)
    # plt.imshow(draw_img)
    # plt.show()
    return draw_img