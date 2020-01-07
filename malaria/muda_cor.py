import cv2
import os
import imutils
import glob
img_dir = "classe_0/" # Enter Directory of all images
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
i = 0
for f1 in files:
    img = cv2.imread(f1)
    #img = imutils.rotate_bound(img, 90)
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite('classe_0/0_' + str(i) + '.jpg',img)
    i+=1
