import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime

frames = 15

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, frames)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, frames)


now = datetime.now() 

color_path = 'video_'+str(now)+'_rgb.avi'
depth_path = 'video_'+str(now)+'_depth.avi'
colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), frames, (640,480), 1)
depthwriter = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'XVID'), frames, (640,480), 1)

pipeline.start(config)


try:
    while True:

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        
        #convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        colorwriter.write(color_image)
        depthwriter.write(depth_colormap)



except KeyboardInterrupt:
    colorwriter.release()
    depthwriter.release()
    pipeline.stop()

