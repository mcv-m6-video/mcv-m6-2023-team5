import cv2
import xmltodict


#Class to represent vehicle detections
class DetectedVehicle:
    def __init__(self, frame, ID, left, top, width, height, conf, right=None, bot=None):
        self.frame = frame
        self.ID   = ID
        self.xtl  = left
        self.ytl  = top
        if right is not None and bot is not None:
            self.xbr  = right
            self.ybr  = bot
            self.w = abs(left - right)
            self.h = abs(top - bot)
        else:
            self.xbr  = left + width
            self.ybr  = top + height
            self.w = width
            self.h = height
        self.conf = conf
    
    def setParked(self, parked):
        self.parked = parked

    def drawRectangleOnImage(self, img, color=(0, 255, 0)):
        cv2.rectangle(img, (self.xtl, self.ytl), (self.xbr, self.ybr), color)

        return img

    def areaOfRec(self):
        return self.w * self.h

    def areaOfIntersection(self, detec2):
        # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
        # indicate top-left and bottom-right corners of the bbox respectively.

        # determine the coordinates of the intersection rectangle
        xA = max(self.xtl, detec2.xtl)
        yA = max(self.ytl, detec2.ytl)
        xB = min(self.xbr, detec2.xbr)
        yB = min(self.ybr, detec2.ybr)
        
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        return interArea


    def areaOfUnion(self, detec2):
        intersectionArea = self.areaOfIntersection(detec2)

        return self.areaOfRec() + detec2.areaOfRec() - intersectionArea


    def IoU(self, detec2):
        return self.areaOfIntersection(detec2) / self.areaOfUnion(detec2)

    def getBBox(self):
        return [self.xtl, self.ytl, self.xbr, self.ybr]

    def __str__(self) -> str:
        return f'Frame {self.frame}, TL [{self.xtl},{self.ytl}], BR [{self.xbr},{self.ybr}], Confidence {self.conf}'
    


def readDetectionsXML(path):
   #Generates detection dictionary where the frame number is the key and the values are the info of the corresponding detections

    with open(path,"r") as xml_obj:
        #coverting the xml data to Python dictionary
        gt = xmltodict.parse(xml_obj.read())
        #closing the file
        xml_obj.close()


    detections = {}
    for track in gt['annotations']['track']:
        #print(track)
        if track['@label'] == 'car':
            for deteccion in track['box']:
                if deteccion['@frame'] not in detections:
                    detections[deteccion['@frame']] = []
                vh = DetectedVehicle(int(deteccion['@frame']), int(track['@id']), 
                                                        float(deteccion['@xtl']), float(deteccion['@ytl']), 0, 0, 
                                                        1.0, float(deteccion['@xbr']), float(deteccion['@ybr']))
                if deteccion['attribute']['@name'] == 'parked' and deteccion['attribute']['#text'] == 'false':
                    vh.setParked(False)
                else:
                    vh.setParked(True)
                detections[deteccion['@frame']].append(vh)

    return detections

def getBBmask(mask):
    counts, hier = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    detectedElems = []
    for cont in counts: #First box is always the background
        x,y,w,h = cv2.boundingRect(cont)

        if 29 < w < 593 and 12 < h < 442: #Condition based on GT minimums* and maximums
            if 0.4 < w/h < 2.5: #Condition to avoid too elongated boxes
                b = DetectedVehicle(0, -1, float(x), float(y), float(w), float(h), float(-1))
                detectedElems.append(b)

    return detectedElems

def drawBBs(image, predictions, color):
    for b in predictions:
        tl = (int(b.xtl), int(b.ytl))
        br = (int(b.xbr), int(b.ybr))
        image = cv2.rectangle(image, tl, br, color, 2)

    return image

def getNotParkedCars(detections):
    notParked = {}
    for frame, objs in detections.items():
        obj_notParked = []
        for ob in objs:
            if not ob.parked:
                obj_notParked.append(ob)
        if len(obj_notParked) > 0:
            notParked[frame] = obj_notParked
    return notParked