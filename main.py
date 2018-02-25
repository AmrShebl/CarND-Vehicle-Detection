from train import *
import cv2
import glob
import matplotlib.pyplot as plt

def main():
    car_detector = CarDetector()
    car_detector.trian_classifier('vehicles','non-vehicles')
    image_files = glob.glob(r'./test_images/test*.jpg')
    subplot_text = '23'
    i=0
    for image_file in image_files:
        i+=1
        img = cv2.imread(image_file)
        draw_image=find_cars(img, clf, scaler)
        plt.subplot(subplot_text+str(i))
        draw_image=cv2.cvtColor(draw_image,cv2.COLOR_BGR2RGB)
        plt.imshow(draw_image)
        plt.title('Example '+str(i))
    plt.show()


if __name__ == "__main__":
    main()