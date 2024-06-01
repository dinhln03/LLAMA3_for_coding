import numpy as np
import os
import cv2
from PIL import Image
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt  # plt 用于显示图片
from tqdm import tqdm

# 标注文件数据处理


def read_pslot(annt_file):
    # print(annt_file)
    with open(annt_file, "r") as f:
        annt = f.readlines()
    # print("annt", annt)
    l = []
    l_ign = []
    for line in annt:
        line_annt = line.strip('\n').split(' ')
        # print(line_annt)

        if len(line_annt) != 13 or line_annt[0] != 'line' or line_annt[-4] == '3':
            continue

        if line_annt[-4] in ['0', '1']:
            l.append(np.array([int(line_annt[i + 1]) for i in range(8)]))
            # continue

        # if line_annt[-4] in ['1', '5']:
        #     l_ign.append(np.array([int(line_annt[i + 1]) for i in range(8)]))
        #     continue
    return l, l_ign

# 标点


def colorize(points_list, img, save_path, item, line, point_color):
    save_path = os.path.join(save_path, str(
        item.strip('.jpg'))+"_"+str(line)+".jpg")
    img2 = img.copy()
    # print(save_path)
    # points_list = 384 * np.abs(np.array(outputs[0], dtype=np.float))
    point_size = 1
    thickness = 4  # 可以为 0、4、8
    for i in range(4):
        cv2.circle(img2, (int(points_list[i][0]), int(points_list[i][1])),
                   point_size, point_color, thickness)
    # print(save_path)
    cv2.imwrite(save_path, img2)

 # 画线


def paint_line(img, dst, cropimg_path, num):
    img2 = img.copy()

    cv2.line(img2, (int(dst[0][0]), int(dst[0][1])), (int(
        dst[1][0]), int(dst[1][1])), (255, 0, 0), 5)
    cv2.line(img2, (int(dst[1][0]), int(dst[1][1])), (int(
        dst[2][0]), int(dst[2][1])), (255, 0, 0), 5)
    cv2.line(img2, (int(dst[2][0]), int(dst[2][1])), (int(
        dst[3][0]), int(dst[3][1])), (255, 0, 0), 5)
    cv2.line(img2, (int(dst[3][0]), int(dst[3][1])), (int(
        dst[0][0]), int(dst[0][1])), (255, 0, 0), 5)

    cropimg_path1 = os.path.join(
        cropimg_path, i.strip('.jpg')+'_'+str(num)+'.jpg')
    cv2.imwrite(cropimg_path1, img2)


def Crop_pic(ps, img_path, cropimg_path, perspective_path, txt_file, i, trans_path, save_path1, save_path2):
    # single pic
    img = cv2.imread(img_path)

    perspective3 = np.float32([[0, 0], [383, 0], [383, 383], [0, 383]])
    perspective3_ = np.float32([[0, 0], [383, 0], [383, 383]])
    num = 0
    for line in ps:
        num = num + 1
        # 随机生成4个坐标
        arr0 = random.randint(80, 120)
        arr1 = random.randint(80, 120)

        arr2 = random.randint(263, 303)
        arr3 = random.randint(80, 120)

        arr4 = random.randint(263, 303)
        arr5 = random.randint(263, 303)

        arr6 = random.randint(80, 120)
        arr7 = random.randint(263, 303)

        perspective0 = np.float32([[line[0], line[1]], [line[2], line[3]], [
            line[4], line[5]], [line[6], line[7]]])
        perspective0_ = np.float32([[line[0], line[1]], [line[2], line[3]], [
            line[4], line[5]]])

        colorize(perspective0, img, save_path1, i, num, (0, 255, 0))

        perspective1 = np.float32(
            [[arr0, arr1], [arr2, arr3], [arr4, arr5], [arr6, arr7]])
        perspective1_ = np.float32(
            [[arr0, arr1], [arr2, arr3], [arr4, arr5]])

        # 求逆变换矩阵
        # trans_inv = cv2.getPerspectiveTransform(perspective1, perspective0)
        trans_inv = cv2.getAffineTransform(perspective1_, perspective0_)

        # 求逆投影变换后的点坐标
        dst = []
        # mat = np.array(
        #     [[[0, 0], [383, 0], [383, 383], [0, 383]]], dtype=np.float32)
        mat = np.array(
            [[0, 0, 1], [383, 0, 1], [383, 383, 1], [0, 383, 1]], dtype=np.float32)
        mat = mat.transpose()
        # dst = cv2.perspectiveTransform(mat, trans_inv)
        dst = np.dot(trans_inv, mat)
        dst = dst.transpose()

        # 画线
        paint_line(img, dst, cropimg_path, num)

        # 将停车位投影变换后得到在384*384分辨率下的停车位图像

        # perspective2 = np.float32([[dst[0][0][0], dst[0][0][1]], [dst[0][1][0], dst[0][1][1]], [
        #                           dst[0][2][0], dst[0][2][1]], [dst[0][3][0], dst[0][3][1]]])
        perspective2_ = np.float32([[dst[0][0], dst[0][1]], [dst[1][0], dst[1][1]], [
            dst[2][0], dst[2][1]]])

        # trans = cv2.getPerspectiveTransform(perspective2, perspective3)
        # dst2 = cv2.warpPerspective(img, trans, (384, 384))

        trans = cv2.getAffineTransform(perspective2_, perspective3_)
        dst2 = cv2.warpAffine(img, trans, (384, 384))

        # 保存原图四个内角点在384*384图片上的坐标
        # mat2 = np.array([[[line[0], line[1]], [line[2], line[3]], [
        #                 line[4], line[5]], [line[6], line[7]]]], dtype=np.float32)
        mat2 = np.array([[line[0], line[1], 1], [line[2], line[3], 1], [
                        line[4], line[5], 1], [line[6], line[7], 1]], dtype=np.float32)

        mat2 = mat2.transpose()
        point = np.dot(trans, mat2)
        point = point.transpose()

        # point = cv2.perspectiveTransform(mat2, trans)
        # point = np.dot(mat2, trans)

        perspective_path1 = os.path.join(
            perspective_path, i.strip('.jpg')+'_'+str(num)+'.jpg')
        # print(perspective_path)
        cv2.imwrite(perspective_path1, dst2)
        colorize(point, dst2, save_path2, i, num, (0, 255, 0))

        # 把四个坐标点记录下来
        txt_file1 = os.path.join(
            txt_file, i.strip('.jpg')+'_'+str(num)+'_OA.txt')
        with open(txt_file1, "w") as f:
            for j in range(4):
                f.write(str(point[j][0]))
                f.write(' ')
                f.write(str(point[j][1]))
                f.write('\n')

        # 把转换矩阵记录下来
        trans_path1 = os.path.join(
            trans_path, i.strip('.jpg')+'_'+str(num)+'.txt')
        with open(trans_path1, "w") as ff:
            for j in range(2):
                for k in range(3):
                    ff.write(str(trans_inv[j][k]))
                    ff.write(" ")

# 计算四个点的预测点与真值点之间的误差


def get_acc(y, y_hat, dis):
    total = 0
    total = 0
    for i in range(4):
        total += ((y[i][0]-y_hat[i][0])**2 + (y[i][1]-y_hat[i][1])**2)**0.5
    total /= 4

    if total < dis:
        return 1
    else:
        return 0


def output_pic(img_path, output_path, trans_path, fina_path, ps2, pix, point_path):
    img_pred = cv2.imread(img_path)
    point_pred = []
    trans_inv = []
    point_pred = np.loadtxt(output_path)

    point_pred = 384*np.expand_dims(point_pred, axis=0)
    trans_inv = np.loadtxt(trans_path)

    trans_inv = trans_inv.reshape(3, 3)
    trans_inv = np.mat(trans_inv)

    point_ground = np.loadtxt(point_path)
    point_ground = np.expand_dims(point_ground, axis=0)
    point_ground2 = cv2.perspectiveTransform(point_ground, trans_inv)
    point_size = 1
    thickness = 4
    for i in range(4):
        cv2.circle(img_pred, (int(point_ground2[0][i][0]), int(point_ground2[0][i][1])),
                   point_size, (0, 255, 0), thickness)

    cv2.imwrite(fina_path, img_pred)

    point_pred2 = cv2.perspectiveTransform(point_pred, trans_inv)

    # 红色
    point_color = (0, 0, 255)
    point_color2 = (0, 255, 0)

    for i in range(4):
        cv2.circle(img_pred, (int(point_pred2[0][i][0]), int(point_pred2[0][i][1])),
                   point_size, point_color, thickness)

    cv2.imwrite(fina_path, img_pred)

    point_pred3 = point_pred2[0]

    ps2 = ps2[0].reshape(4, 2)

    tmp = get_acc(point_pred3, point_ground2[0], pix)
    return tmp

# 精度


def output(pix):
    accuracy = 0
    for i in os.listdir(test_dir):
        output_path = os.path.join(
            "/media/alpha4TB/ziqi/Parking/CNN/output", i.strip('.jpg')+'.txt')
        img_path = os.path.join(
            "/media/alpha4TB/ziqi/Parking/Ps_locate_dataset/img", i)
        trans_inv = os.path.join(
            "/media/alpha4TB/ziqi/Parking/Ps_locate_dataset/trans_inv", i.strip('.jpg')+'.txt')
        fina_path = os.path.join(
            "/media/alpha4TB/ziqi/Parking/Ps_locate_dataset/fina", i)
        annt_path2 = os.path.join(
            './Ps_locate_dataset/annt', i.strip('.jpg')+'_OA.txt')
        point_path = os.path.join(
            "/media/alpha4TB/ziqi/Parking/Ps_locate_dataset/point", i.strip('.jpg')+'_OA.txt')
        # print(fina_path)
        ps2, _ = read_pslot(annt_path2)
        tmp = output_pic(img_path, output_path,
                         trans_inv, fina_path, ps2, pix, point_path)
        accuracy += tmp
    return accuracy


if __name__ == "__main__":
    data_dir = '/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/pic'
    label_dir = '/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/annotation'
    crop_dir = '/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/crop_img'
    perspective_dir = '/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/perspective_img'
    txt_dir = '/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/point'
    cnt = 0
    f1 = open(
        "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/train_list.txt", "w")
    # f2 = open("./Ps_locate_dataset/val_list.txt", "w")
    test_dir = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/test_img"
    trans_path = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/trans_inv"
    save_path1 = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/src_img"
    save_path2 = "/media/home_bak/ziqi/park/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/perspective2_img"

    pbar = tqdm(total=len(os.listdir(data_dir)))

    for i in os.listdir(data_dir):
        # print(i)
        annt_file = os.path.join(label_dir, i.strip('.jpg')+'_OA.txt')
        img_path = os.path.join(data_dir, i)

        ps, _ = read_pslot(annt_file)
        Crop_pic(ps, img_path, crop_dir,
                 perspective_dir, txt_dir, i, trans_path, save_path1, save_path2)
        pbar.update(1)

    pbar.close()

    # acc = []
    # for k in range(31):
    #     print("k", k)
    #     x1 = output(k)
    #     x1 = 100 * x1 / 743
    #     acc.append(x1)

    # x1 = round(x1, 3)
    # print(acc)
    # print(len(acc))

    # # 设置画布大小
    # plt.figure(figsize=(30, 15))

    # # 标题
    # plt.title("accruracy distribution")

    # # 数据
    # plt.bar(range(len(acc)), acc)

    # # 横坐标描述
    # plt.xlabel('pixel')

    # # 纵坐标描述
    # plt.ylabel('accuracy')
    # # # 设置数字标签
    # # for a, b in zip(x, acc):
    # #     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

    # plt.savefig(
    #     "/media/alpha4TB/ziqi/Parking/Ps_locate_dataset/PLD_BirdView_TrainingDaraSet_All/accuracy.png")

    # 保存训练数据的文件名
    filenames = os.listdir(perspective_dir)
    filenames.sort()
    print(filenames[0])

    for i in os.listdir(perspective_dir):
        perspective_path = os.path.join(perspective_dir, i)
        f1.write(perspective_path)
        f1.write('\n')

    f1.close()
