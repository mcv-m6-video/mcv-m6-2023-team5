Task 1: IoU,AP,mAP.
Task 2: Temporal analysis of the results (MIoU over Time (frame)).
Task 3: Quantitatide evaluation of optical flow.
Task 4:
1. Decode 16-bit .png OF representation with the "get_offset" function
2. Compute magnitude (l2-norm) and angle (atan2) of OF
3. Limit maximum magnitude and normalize it. Good bound values are typically around 3. The less the bound -- the more compressed the OF is. The bigger the bound -- more information, but darker image.
4. Create an HSV image, assigning:
    - Hue value to the direction of OF;
    - Setting maximum saturation to all the pixels
    - Value (brightness) as the OF magnitude
RESULT: Different colors represent different directions of OF.
The brighter the pixels -- the stronger OF was at this point.
    
