import time

import cv2
import numpy as np
j = 1
while 1:

    path = 'Bearing/' + str(j) + '.jpg'
    img = cv2.imread(path)
    img_copy = img.copy()
    img = cv2.blur(img, (1, 1))
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    # flag, img_copy = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    imgray = cv2.Canny(img_copy, 600, 100, 3)  # Canny边缘检测，参数可更改

    # cv2.imshow("imgray",imgray)
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("thresh", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # contours为轮廓集，可以计算轮廓的长度、面积等
    ux = 0
    uy = 0
    for cnt in contours:
        if len(cnt) > 50:
            # S1 = cv2.contourArea(cnt)  # 格林公式计算的实际面积
            ell = cv2.fitEllipse(cnt)  # 拟合椭圆 ellipse = [ center(x, y) , long short (a, b), angle ]
            x = int(ell[0][0])
            y = int(ell[0][1])
            a = ell[1][0]
            b = ell[1][1]
            # S2 = math.pi * ell[1][0] * ell[1][1]  # 理论面积
            if (b / a) < 1.2:  # and a > 0 and b > 0 and a < 0 and b < 0:  # 面积比例
                uy = y
                ux = x
                img = cv2.ellipse(img, ell, (0, 0, 200), 2)
                cv2.circle(img, (x, y), 2, (255, 255, 255), 3)
                cv2.putText(img, str((x, y)), (x + 20, y + 10), 0, 0.5,
                            [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                print("长轴： " + str(a) + "    " + "短轴： " + str(b) + "   " + str(ell[0][0]) + "   " + str(ell[0][1]))
    cv2.imshow("ell", img)
    j+=1
    if j==44:
        j=1

    time.sleep(0.5)
    cv2.waitKey(20)

