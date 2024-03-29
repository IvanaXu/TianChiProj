import kfbReader
import cv2 as cv


n = "53"
path = "../data/T2019_"+n+".kfb"
scale = 20

read = kfbReader.reader()
# ReadInfo
read.ReadInfo(path, scale, False)

# roi = read.ReadRoi(10240, 10240, 512, 512, scale)
# cv.imshow("roi", roi)
# cv.waitKey(0)

roi = read.ReadRoi(10240, 10240, 512, 512, scale)
# cv.imshow("roi", roi)
cv.imwrite("../out/roi.jpg", roi)

h, w, s = read.getHeight(), read.getWidth(), read.getReadScale()
print(h, w, s)

# read.SetReadScale(scale=20)

roi_0 = read.ReadRoi(5120,5120,256,256,scale=5)
# cv.imshow("roi", roi_0)
cv.imwrite("../out/roi_0.jpg", roi_0)

roi_1 = read.ReadRoi(10240,10240,512,512,scale=10)
# cv.imshow("roi", roi_1)
cv.imwrite("../out/roi_1.jpg", roi_1)

roi_2 = read.ReadRoi(20480,20480,1024,1024,scale=20)
# cv.imshow("roi", roi_2)
cv.imwrite("../out/roi_2.jpg", roi_2)

roi_r = read.ReadRoi(5596,19265,11168,6887,scale=20)
cv.imwrite("../out/roi_r.jpg", roi_r)

roi_r1 = read.ReadRoi(13163,21440,143,154,scale=20)
cv.imwrite("../out/roi_r1.jpg", roi_r1)

roi_r2 = read.ReadRoi(6827,22658,201,159,scale=20)
cv.imwrite("../out/roi_r2.jpg", roi_r2)







