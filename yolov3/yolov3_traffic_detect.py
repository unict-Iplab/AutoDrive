# -*- coding: utf-8 -*-
from ctypes import *
import math
import random
import cv2
fx= 3150.0    #fx=f/dx
fy= 3500.0   #fy=f/dy
c=960.0
d=540.0
h=5.5
import math
a = math.atan(1/6.0)
tan = math.tan(a)
cos = math.cos(a)
sin= math.sin(a)
def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
#结果可视化
def vis_detections(im, class_name, dets, thresh=0.5):
    #通过阈值过滤选框
    inds = np.where(dets[:, -1] >= thresh)[0]
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
	#print bbox[0:4]
	x = bbox[0]
	y = bbox[1]
	x2 = bbox[2]
	y2 = bbox[3]
	
	color = (0,0,255)
	if(class_name == 'person'):
		color = (255,0,0)
	elif(class_name == 'bus'):
		color = (120,120,0)
	elif(class_name == 'car'):
		color = (0,255,0)
	else:
		continue
	u = (x + x2) / 2
	v = y2
	z1= (h*fy+h*tan*(d-v))/(v-d+fy*tan)
        x1 = (z1 * cos + h * sin) * (u - c) / fx
	cv2.putText(frame, '[{:.1f} {:.1f}]'.format(x1,z1), (int(x + 2), int(y2 - 4)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	cv2.putText(frame, '{:s} {:.3f}'.format(class_name, score), (int(x), int(y-2)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	cv2.rectangle(frame, (x, y), (x2, y2), color, 2)					#绘制范围边框
	
    cv2.imshow("Video",im)


#图像检测流程4096
def demo(net, meta, path):
    r = detect(net, meta, path)
    CONF_THRESH = 0.3
    NMS_THRESH = 0.3
    for a in r:
	x = a[2][0]
	y = a[2][1]
	x2 = a[2][2] + x
	y2 = a[2][3] + y
	color = (0,0,255)
	if(class_name == 'person'):
		color = (255,0,0)
	elif(class_name == 'bus'):
		color = (120,120,0)
	elif(class_name == 'car'):
		color = (0,255,0)
	u = (x + x2) / 2
	v = y2
	z1= (h*fy+h*tan*(d-v))/(v-d+fy*tan)
        x1 = (z1 * cos + h * sin) * (u - c) / fx
	cv2.putText(frame, '[{:.1f} {:.1f}]'.format(x1,z1), (int(x + 2), int(y2 - 4)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	cv2.putText(frame, '{:s} {:.3f}'.format(class_name, score), (int(x), int(y-2)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
    
if __name__ == "__main__":

    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    set_gpu(0)
    net = load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
    meta = load_meta("cfg/coco.data")
    a = 1060
    while(True):
	    path = str(a) +'.jpg'
    	    r = detect(net, meta, path)
   
	    frame = cv2.imread(path)
	    for b in r:
		x = b[2][0] - b[2][2] / 2
		y = b[2][1] - b[2][3] / 2
		x2 = b[2][2] + x
		y2 = b[2][3] + y
		color = (0,0,255)
		if(b[0] == 'person'):
			color = (255,0,0)
		elif(b[0] == 'bus'):
			color = (120,120,0)
		elif(b[0] == 'car'):
			color = (116,139,69)
		u = (x + x2) / 2
		v = y2
		z1= (h*fy+h*tan*(d-v))/(v-d+fy*tan)
		x1 = (z1 * cos + h * sin) * (u - c) / fx
		cv2.putText(frame, '[{:d} {:d}]'.format(int(x1),int(z1)), (int(x + 2), int(y - 3)),cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
		#cv2.putText(frame, '{:.3f}'.format(b[1]), (int(x), int(y-2)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), color, 4)
	    cv2.imwrite(path,frame)
	    cv2.imshow("show",frame)
	    print (a)
	    #cv2.waitKey(0)
	    a = a + 1
    	#print (r[0][0])
    	#print (r[0][1])
    	#print (r[0][2][0])
    	#print (r[0][2][1])
    	#print (r[0][2][2])
    	#print (r[0][2][3])
	    if a > 1118:
	    	break
    
    

