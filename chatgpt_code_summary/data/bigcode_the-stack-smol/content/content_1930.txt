from hog_cpp.fhog import fhog
import numpy as np

'''
https://github.com/lawpdas/fhog-python
'''

def get_hog(img):
    M = np.zeros(img.shape[:2], dtype='float32')
    O = np.zeros(img.shape[:2], dtype='float32')
    H = np.zeros([img.shape[0] // 4, img.shape[1] // 4, 32], dtype='float32')  # python3
    fhog.gradientMag(img.astype(np.float32), M, O)
    fhog.gradientHist(M, O, H)
    H = H[:, :, :31]

    return H



'''
if __name__ == "__main__":
    img_path = 'D:/DATASET/OTB100/Basketball/img/0001.jpg'
    img = cv2.imread(img_path)

    sub = img[0:40, 0:40]

    H = get_hog(sub)
    print(H)
'''