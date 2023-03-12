import cv2
import utils as u
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

green=(0, 255, 0)
red=(255, 0, 0)
blue=(0, 0, 255)
#get the gt and the detector Bboxes
vec_list, obj_list=u.reader("/home/michell/Documents/M6/AICity_data/AICity_data/train/S03/c010/gt/gt.txt")
vec_list_yolo, obj_list_yolo=u.reader("/home/michell/Documents/M6/AICity_data/AICity_data/train/S03/c010/det/det_yolo3.txt")
vec_list_ssd512, obj_list_ssd512=u.reader("/home/michell/Documents/M6/AICity_data/AICity_data/train/S03/c010/det/det_ssd512.txt")
vec_list_rcnn, obj_list_rcnn=u.reader("/home/michell/Documents/M6/AICity_data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt")

#grouped
od=u.group(obj_list)
od_yolo=od=u.group(obj_list_yolo)
od_ssd512=od=u.group(obj_list_ssd512)
od_rcnn=od=u.group(obj_list_rcnn)


# Open video file
cap = cv2.VideoCapture("/home/michell/Documents/M6/AICity_data/AICity_data/train/S03/c010/vdo.avi")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = 0
# Loop through frames
frames_gif = []
while True:
    # Read frame
    ret, frame = cap.read()
    # Check if frame was successfully read
    if not ret:
        break
    # Frame counter 
    frame_count += 1
    # Display frame
    
    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    #display gt BBoxes
    u.drawBB(frame,frame_num,od,green,0)
    #yolo
    u.drawBB(frame,frame_num,od_yolo,red,0.9)
    #ssd512
   # u.drawBB(frame,frame_num,od_ssd512,red,0.9)
    #rcnn
   # u.drawBB(frame,frame_num,od_rcnn,red,0.9)
    frame = cv2.resize(frame, (int(width/6), int(height/6)))
    # Save the frame fot the gif each X frames in order  To make it less heavy 
    if frame_count % 5 == 0:
        frames_gif.append(frame)
        
    cv2.imshow("Frame", frame)

    # Check for quit command
    if cv2.waitKey(1) == ord('q'):
        break

# Save the video frames as a GIF
imageio.mimsave('output.gif', frames_gif, duration=0.05)

# Create la gift
ani = FuncAnimation(plt.gcf(), u.update, frames=10, interval=200)

# Save gif
ani.save('animacion.gif', writer='imagemagick')

# Release video file and close windows
cap.release()
cv2.destroyAllWindows()
