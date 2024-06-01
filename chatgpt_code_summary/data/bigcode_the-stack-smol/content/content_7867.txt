import cv2
import glob
import os


if __name__ == "__main__":
    img_path = "/workspace/mnt/storage/yankai/test_cephfs/YOLOX/datasets/ducha_det/labels/val"
    save_size_path = "/workspace/mnt/storage/yankai/test_cephfs/YOLOX/datasets/ducha_det/sizes/val"
    if not os.path.exists(save_size_path):
        os.makedirs(save_size_path)
    img_labels = glob.glob(img_path + "/*.txt")
    for i, img_label in enumerate(img_labels):
        if i % 100 == 0:
            print(i)
        img_path = img_label.replace('labels','images').replace('.txt','.jpg').replace('.txt','.png').replace('.txt','.jpeg')
        assert os.path.exists(img_path)
        img = cv2.imread(img_path)
        assert img is not None
        h, w, _ = img.shape
        save_size_path = img_label.replace('labels', 'sizes')
        with open(save_size_path, 'w') as txt_write:
            txt_write.writelines("{} {}".format(h, w))
