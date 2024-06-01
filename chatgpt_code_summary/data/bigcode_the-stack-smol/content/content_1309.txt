import cv2
from PIL import ImageGrab
import numpy as np


def main():
    while True:
        # bbox specifies specific region (bbox= x,y,width,height)
        img = ImageGrab.grab(bbox=(0, 40, 1075, 640))
        vanilla = img_np = np.array(img)
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image = cv2.drawContours(img_np, contours, -1, (0, 255, 0), 2)
        cv2.imshow("test", image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            print("test")
            break
        else:
            cv2.waitKey(1)
            # cv2.waitKey(0)


if __name__ == "__main__":
    main()
