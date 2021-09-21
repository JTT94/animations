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
    while still_reading:
        fp = os.path.join(out_dir, f"frame_{frame_count:03d}.jpg")
        cv2.imwrite(fp, image)
        
        # read next image
        still_reading, image = video_capture.read()
        frame_count += 1




def make_gif(frame_folder, name="test"):
    images = glob.glob(f"{frame_folder}/*.jpg")
    images.sort()
    frames = [Image.open(image) for image in images]
    frame_one = frames[0]
    frame_one.save(f"{name}.gif", format="GIF", append_images=frames,
                   save_all=True, duration=50, loop=0)


if __name__ == "__main__":
    out_dir = 'output'
    
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    path =  sys.argv[1] #'/data/hylia/thornton/animations/media/videos/barchart/1080p60/BarChartExample.mp4'  
    convert_mp4_to_jpgs(path,out_dir)
    make_gif(out_dir)
    #shutil.rmtree(out_dir)


