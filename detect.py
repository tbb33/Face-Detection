#detect face in img using Cascade Classifier object
import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("photo.jpg")
grey_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#smaller scalerfactor is more accurate but takes more time
faces = face_cascade.detectMultiScale(grey_img, scaleFactor=1.1,
minNeighbors=5)

#draw rectangle in new img
for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),5)

resized = cv2.resize(img,(int( img.shape[1]/3), int( img.shape[0]/3) ))

cv2.imshow("GreyImg",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
