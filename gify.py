import glob
import os
import moviepy.video.io.ImageSequenceClip

images= glob.glob(f"results/120fps/*")
image_files = sorted(images, key = lambda fp : float(fp.split('\\')[-1][:-4]))

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=60)
clip.write_videofile('120fps.mp4')

