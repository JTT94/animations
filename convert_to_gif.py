import shutil
import cv2
import os
import sys
import glob
from PIL import Image

def convert_mp4_to_jpgs(path, out_dir):
    video_capture = cv2.VideoCapture(path)
    still_reading, image = video_capture.read()
    frame_count = 0
    fp_count = 0
    while still_reading:
        if frame_count % 3 ==0 :
            fp = os.path.join(out_dir, "frame_{0}.jpg".format(fp_count))
            cv2.imwrite(fp, image)
            fp_count+=1
        
        # read next image
        still_reading, image = video_capture.read()
        frame_count += 1




def make_gif(frame_folder, name="test"):
    #images = glob.glob(f"{frame_folder}/*.jpg")
    images = os.listdir(frame_folder)
    N = len(images)
    images = [os.path.join(frame_folder, "frame_{0}.jpg".format(i)) for i in range(N)]
    frames = [Image.open(image) for image in images]
    frame_one = frames[0]
    w,h = frame_one.size
    # left upper, right lower
    frames = [f.crop((30, 0, w-30, h)) for f in frames ]
    frame_one.save(f"{name}.gif", format="GIF", append_images=frames,
                   save_all=True, duration=20, loop=0)


if __name__ == "__main__":
    out_dir = 'output'
    
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    path =  sys.argv[1] #'/data/hylia/thornton/animations/media/videos/barchart/1080p60/BarChartExample.mp4'  
    convert_mp4_to_jpgs(path,out_dir)
    make_gif(out_dir)
    #shutil.rmtree(out_dir)


