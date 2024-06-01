import cv2
import numpy as np
from PIL import Image


def draw_approx_polyDP(cnt, epsilon=0.01, closed=True):
    """用多边形来近似的表示曲线"""
    epsilon = epsilon * cv2.arcLength(cnt, closed)  # 得到轮廓的周长信息作为参考值
    return cv2.approxPolyDP(cnt, epsilon, closed)  # 得到近似多边形框


def draw_convex_hull(cnt):
    """画凸包，传入的是一些点"""
    return cv2.convexHull(cnt)  # 获取处理过的轮廓信息


def show_img(file_name, window_name='win'):
    img = cv2.imread(file_name)
    cv2.imshow(window_name, img)
    # 按任意键，图片消失
    cv2.waitKey()
    cv2.destroyAllWindows()


def camera_show(window_name='camera'):
    """最好在改进一下关闭窗口部分的功能
    建立一个窗口捕捉摄像头显示的内容
    当左键点击过窗口，且按过任意键盘键，才会退出窗口"""
    clicked = False
    camera_capture = cv2.VideoCapture(0)

    def on_mouse(event, x, y, flags, param):
        global clicked
        if event == cv2.EVENT_LBUTTONUP:
            clicked = True

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    success, frame = camera_capture.read()
    # cv2.waitKey(1) 参数表示等待键盘触发的时间，返回值为-1表示没有见按下
    while success and cv2.waitKey(1) == -1 and not clicked:
        cv2.imshow(window_name, frame)
        success, frame = camera_capture.read()
    cv2.destroyAllWindows()
    camera_capture.release()


def camera_save(file_name, seconds=3, fps=60):
    # 获得设备
    camera_capture = cv2.VideoCapture(0)
    size = (int(camera_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(camera_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    video_writer = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

    success, frame = camera_capture.read()
    num_frames_remaining = seconds * fps - 1
    while success and num_frames_remaining > 0:
        video_writer.write(frame)
        success, frame = camera_capture.read()
        num_frames_remaining -= 1
    camera_capture.release()


def copy(orig_img, start_height, start_width, part):
    height, width = part.shape
    orig_img[start_height: start_height + height, start_width: start_width + width] = part
    return orig_img


def draw_gray_random(height, width):
    flat_numpy_array = np.random.randint(0, 256, height * width)
    gray_image = flat_numpy_array.reshape(height, width)
    return gray_image


def draw_random(height, width, channel=3):
    flat_numpy_array = np.random.randint(0, 256, height * width * channel)
    bgr_image = flat_numpy_array.reshape((height, width, channel))
    return bgr_image


def draw_gray_black(height, width):
    img = np.zeros((height, width), dtype=np.uint8)
    return img


def draw_line(img, x1, y1, x2, y2, color=(0, 255, 0), thickness=2):
    return cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_rectangle(img, box, contour_idx=0, color=(0, 0, 255), thickness=3):
    return cv2.drawContours(img, box, contour_idx, color, thickness)


def draw_cicile(img, center, radius, color=(0, 255, 0), thickness=2):
    return cv2.circle(img, center, radius, color, thickness)


def draw_black(height, width):
    img = draw_black(height, width)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def img2array(img):
    return bytearray(img)


def array_img(arr, height, width, channel=3):
    return np.array(arr).reshape(height, width, channel)


def array2img_gray(arr, height, width):
    return np.array(arr).reshape(height, width)


if __name__ == '__main__':

    img = cv2.imread('sphere.png')
    cv2.imshow('win', img)
    # empire = Image.open('sphere.png')
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # print(empire.shape())
    # empire.convert('RGB')
    # print(empire.mode)
    # print(empire.shape())

    img = Image.open('sphere.png')
    img = img.resize((137, 137))
    # 将黑色的部分变为透明
    print(img.info)
    print(img.mode)
    img = img.convert("RGBA")
    print(img.mode)
    width = img.size[0]
    height = img.size[1]
    for x in range(width):
        for y in range(height):
            r, g, b, a = img.getpixel((x, y))
            rgba = (r, g, b, a)
            if (r == g == b == 0):
                img.putpixel((x, y), (0, 0, 0, 0))
    img.save('sphere_2.png')
    img.show()
