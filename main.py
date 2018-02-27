from train import *
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip, ImageSequenceClip

def main():
    car_detector = CarDetector(5)
    car_detector.trian_classifier('vehicles','non-vehicles')
    # image_files = glob.glob(r'./test_images/test*.jpg')
    # subplot_text = '23'
    # i=0
    # for image_file in image_files:
    #     i+=1
    #     img = cv2.imread(image_file)
    #     draw_image= car_detector.find_cars(img)
    #     plt.subplot(subplot_text+str(i))
    #     draw_image=cv2.cvtColor(draw_image,cv2.COLOR_BGR2RGB)
    #     plt.imshow(draw_image)
    #     plt.title('Example '+str(i))
    # plt.show()

    input_clip_file = 'project_video.mp4'
    input_clip = VideoFileClip(input_clip_file)
    output_clip_file = 'output_project_video.mp4'
    output_frame_list = []
    for frame in input_clip.iter_frames():
        #The algorithms were developed for BGR not RGB
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        output_frame = car_detector.find_cars(frame)
        output_frame = cv2.cvtColor(output_frame,cv2.COLOR_BGR2RGB)
        output_frame_list.append(output_frame)

    output_clip = ImageSequenceClip(output_frame_list, input_clip.fps)
    output_clip.write_videofile(output_clip_file)


if __name__ == "__main__":
    main()