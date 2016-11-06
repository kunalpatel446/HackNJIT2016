import cv2
import numpy as np
import sys

# functions for program

# Downsample an image to have no more than the specified maximum height
def downsample(src, hmax):
    h, w = src.shape[:2]
    while h > hmax:
        h /= 2
        w /= 2
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)

# find center of rectangle
def rect_center(rect):
    x,y,w,h = rect
    return np.array( (x+0.5*w, y+0.5*h) )

# scale a rectangle
def rect_scale(rect,scl):
    return np.array(rect)*scl

##########################################################
# Main function for face-swapping

# Hit the 'b' key to toggle blurring
do_eye = False

# Load our cascade classifiers
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
mouth_detector = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

# We will either use a VideoCapture object or a static image
cap = None
static_image = cv2.imread('lena.jpg')

# You can specify a video device (e.g. 0) on the command line, a
# movie file, or a static image filename
if len(sys.argv) == 2:
    try:
        device_num = int(sys.argv[1])
        cap = cv2.VideoCapture(device_num)
    except:
        src_file = sys.argv[1]
        static_image = cv2.imread(src_file)
        if static_image is None or not len(static_image):
            cap = cv2.VideoCapture(0)
        else:
            do_eye = 1

cascades = [ cv2.CascadeClassifier(arg) for arg in sys.argv[2:] ]

# colors for rectangles drawings
colors = [
          (  0,   0, 255),
          (  0, 255, 255),
          (  0, 255,   0),
          (255, 255,   0),
          (255,   0,   0),
          (255,   0,  255)
          ]

# Create our window
cv2.namedWindow('win')

# initialize variables
face_rect1 = None
face_rect2 = None
eye_rect1 = None
eye_rect2 = None
rects= None
e_rects = None
input_images = None
be1=None
te1=None
te2=None
be2=None
alpha=0.5

# Main loop
while True:
    # Pull an image from capture or the static image
    if cap is None:
        img = static_image.copy()
    else:
        ok, img = cap.read()
        if not ok or img is None:
            break
            
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Downsample for faster runtime -- this is important!
    img_small = downsample(img_gray, 320)

    # Figure out what scale factor we downsampled by - we will need it
    # to rescale the rectangles returned by the cascade.
    scl = img.shape[0] / img_small.shape[0]

    # Do histogram equalization - should help a bit with detection
    img_eq = cv2.equalizeHist(img_small)

    # Run the cascade classifier
    face_rects = np.array(face_detector.detectMultiScale(img_eq))

    # Face classification, tracking, and interpolation:
    # if we find any faces
    if len(face_rects):
    """    if face_rect2 is None:
            if face_rect1 is None:
                # if first detection, just choose largest area
                areas = face_rects[:,2] * face_rects[:,3]
                face_rect1 = face_rects[areas.argmax()] # find one w/ largest area
            else:
                # if second detection, just choose smallest area
                areas = face_rects[:,2] * face_rects[:,3]
                face_rect2 = face_rects[areas.argmin()] # find one w/ smallest area"""

        #else:
            # otherwise, choose one closest to prev detection
            # first face
        face_rect1 = face_rects[areas.argmax()]
        face_center1 = rect_center(face_rect1)  # face center of previous detection
        centers1 = face_rects[:,:2] + 0.5*face_rects[:,2:] # centers of current faces
        diffs1 = centers1 - face_center1      # difference of currents and previous
        p1 = face_rects[(diffs1**2).sum(axis=1).argmin()] # minimum of all diffs

            # Second face
            #face_center2 = rect_center(face_rect2)
            #centers2 = face_rects[:,:2] + 0.5*face_rects[:,2:]
            #diffs2 = centers2 - face_center2
            #p2 = face_rects[(diffs2**2).sum(axis=1).argmin()]
            
            # Check if faces are the same. Otherwise, do linear interpolation to
            # keep faces consistently tracked
            #if (np.array_equal(p1,p2)):
        rect = np.array([p1])
        cv2.rectangle(im)
            #else:
                #face_rect1 = alpha*p1 + (1-alpha)*face_rect1
                #face_rect2 = alpha*p2 + (1-alpha)*face_rect2
                #rects = np.array([face_rect1,face_rect2])


    # Face and Eye dimension-saving process:
    # If we have two faces, find dimensions of face rectangle for later face swapping.
    # Also detect eyes and find dimensions for later eye swapping.
    if rects is not None:
        xr, yr, wr, hr = tuple(rect.astype(int))
        xr *= scl
        yr *= scl
        wr *= scl
        hr *= scl

        # get subimage in ROI
        subimg = img[yr:yr+hr, xr:xr+hr]
        subimg_gray = img_gray[yr:yr+hr, xr:xr+hr]
        
        # Downsample for faster runtime -- this is important!
        subimg_small = downsample(subimg_gray, 320)
            
        # Figure out what scale factor we downsampled by - we will need it
        # to rescale the rectangles returned by the cascade.
        e_scl = subimg.shape[0] / subimg_small.shape[0]

        # find eyes
        eye_rects = eye_detector.detectMultiScale(subimg_gray)

        # Eye classification and tracking:
        # if we find any eyes
        if len(eye_rects):
            if eye_rect2 is None:
                if eye_rect1 is None:
                    # if first detection, just choose largest area
                    areas = eye_rects[:,2] * eye_rects[:,3]
                    eye_rect1 = eye_rects[areas.argmax()] # find one w/ largest area
                else:
                    # if second detection, just choose smallest area
                    areas = eye_rects[:,2] * eye_rects[:,3]
                    eye_rect2 = eye_rects[areas.argmin()] # find one w/ smallest area
                            
            else:
                # otherwise, choose one closest to prev detection
                # first eye
                eye_center1 = rect_center(eye_rect1)
                centers1 = eye_rects[:,:2] + 0.5*eye_rects[:,2:]
                diffs1 = centers1 - eye_center1
                eye_rect1 = eye_rects[(diffs1**2).sum(axis=1).argmin()]
                                
                # second eye
                eye_center2 = rect_center(eye_rect2)
                centers2 = eye_rects[:,:2] + 0.5*eye_rects[:,2:]
                diffs2 = centers2 - eye_center2
                eye_rect2 = eye_rects[(diffs2**2).sum(axis=1).argmin()]
                
                # check if eyes are the same for looping later
                if (np.array_equal(eye_rect1,eye_rect2)):
                    e_rects = np.array([eye_rect1])
                    eye_rect2 = None
                else:
                    e_rects = np.array([eye_rect1,eye_rect2])

        # eye accumulator
        j = 0
        
        # Eye dimension saving:
        # If we have >1 eye(s) for the particular face, find dimensions of eyes for later
        # eye swapping.
        if e_rects is not None:
            
            # Loop through each eye
            for eye_rect in e_rects:
                xe,ye,we,he = eye_rect
                xe *= e_scl
                ye *= e_scl
                we *= e_scl
                he *= e_scl
                
                # Save individual eye subimage
                subimge = subimg[ye:ye+he, xe:xe+he]
                
                # If statements save dimensions for each eye in each face.
                # Saved as different variables to keep track of all subimages and corresponding
                # dimensions
                if i==0:
                    # first face
                    if j==0:
                        #first eye
                        te1 =subimge
                        tex1 = xe
                        tey1 = ye
                        teh1 = he
                        tew1 = we
                        tesize1 = (tew1,teh1)
                    else:
                        # second eye
                        te2=subimge
                        tex2 = xe
                        tey2 = ye
                        teh2 = he
                        tew2 = we
                        tesize2 = (tew2,teh2)
                else:
                    # second face
                    if j==0:
                        # first eye
                        be1 =subimge
                        bex1 = xe
                        bey1 = ye
                        beh1 = he
                        bew1 = we
                        besize1 = (bew1,beh1)
                    else:
                        # second eye
                        be2=subimge
                        bex2 = xe
                        bey2 = ye
                        beh2 = he
                        bew2 = we
                        besize2 = (bew2,beh2)
                j = j+1

        # Face dimension saving:
        # save dimensions for first face
        if i==0:
            t = subimg
            tx = xr
            ty = yr
            th = subimg.shape[0]
            tw = subimg.shape[1]
            tsize = (tw,th)
                
        # save dimensions for second face
        else:
            b=subimg
            bx = xr
            by = yr
            bh = subimg.shape[0]
            bw = subimg.shape[1]
            bsize = (bw,bh)
        i = i+1

    # Show it
    cv2.imshow('win', img)

    # Handle key presses - 'b' toggles eye swapping
    k = cv2.waitKey(5)
    if k == 27:
        break
    elif k == ord('b'):
        do_eye = not do_eye
