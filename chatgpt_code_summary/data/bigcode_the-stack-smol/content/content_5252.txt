import argparse
import glob
import os
import time
import vlc

import cv2
import numpy as np
from enum import Enum
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from align.align_trans import get_reference_facial_points
from align.detector import load_detect_faces_models, process_faces
from align.visualization_utils import draw_fps, show_results
from util.extract_feature_v2 import extract_feature_for_img, load_face_id_model

MIN_FACE_PROB = 0.9
STREAM_DIR = '/home/ec2-user/projects/facelab-data/stream-data'
RESULT_DIR = '/home/ec2-user/projects/facelab-data/results'
ID_FEATURES_DIR = '/home/ec2-user/projects/facelab-data/test_Aligned/'
FACE_ID_MODEL_ROOT = '/home/ec2-user/projects/facelab-data/models/backbone_ir50_ms1m_epoch120.pth'
FONT_PATH = '/usr/share/fonts/dejavu/DejaVuSans.ttf'


class Mode(Enum):
    DEMO = 1
    FILE = 2

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Mode[s]
        except KeyError:
            raise ValueError()


def load_id_files(id_features_dir):
    id_npy = {}
    for path in glob.glob('{}/*.npy'.format(id_features_dir)):
        name = os.path.splitext(os.path.basename(path))[0]
        id_npy[name] = np.load(path)
    return id_npy


def check_identity(id_npy, query_features, max_min_dist=1.0):
    distances_from_id = {}
    for name, id_npy_arr in id_npy.items():
        distances_from_id[name] = []
        for id_npy_row in id_npy_arr:
            dist = np.linalg.norm(id_npy_row - query_features)
            distances_from_id[name].append(dist)

    min_dist = np.finfo(float).max
    name_match = ''
    for name, distances in distances_from_id.items():
        avg = np.mean(distances)
        if avg < min_dist:
            min_dist = avg
            name_match = name

    if min_dist > max_min_dist:
        name_match = 'Unknown'

    return name_match, min_dist


def process_and_viz_img(pil_img,
                        det_models,
                        face_id_model,
                        reference,
                        crop_size,
                        id_npy,
                        font):
    # Detect bboxes and landmarks for all faces in the image and warp the
    # faces.
    face_results = process_faces(
        img=pil_img,
        det_models=det_models,
        reference=reference,
        crop_size=crop_size)

    # Filter results by detection probability.
    filtered_face_results = []
    for face_result in face_results:
        face_prob = face_result.bounding_box[4]
        if face_prob < MIN_FACE_PROB:
            print('Skipping detection with low face probability: {:.2f}'.format(face_prob))
            continue

        filtered_face_results.append(face_result)

    face_results = filtered_face_results

    identity_list = []
    for face_result in face_results:
        features = extract_feature_for_img(
            img=face_result.warped_face,
            backbone=face_id_model)
        # features is tensor, so converting to numpy arr below
        identity, min_dist = check_identity(
            id_npy=id_npy,
            query_features=features.numpy())
        identity_list.append((identity, '({:.2f})'.format(min_dist)))

    # Visualize the results
    viz_img = show_results(
        img=pil_img,
        bounding_boxes=[
            fr.bounding_box
            for fr in face_results
        ],
        facial_landmarks=[
            fr.landmark
            for fr in face_results
        ],
        names=identity_list,
        font=font)

    if identity_list:
        names = list(zip(*identity_list))[0]
    else:
        names = []
    return viz_img, names


def play_sound_for_name(name):
    name_to_sound_file = {
        'neelam': '/Users/bkovacs/Documents/neelam-how-is-it-going.m4a',
        'kovi': '/Users/bkovacs/Documents/balazs-how-is-it-going.m4a',
    }
    name = name.lower()
    if name not in name_to_sound_file:
        return
    player = vlc.MediaPlayer(name_to_sound_file[name])
    player.play()


def play_sound_if_needed(names,
                         name_to_last_time_seen,
                         cur_time,
                         min_elapsed_to_play=3):
    for name in names:
        if (name not in name_to_last_time_seen or
                name_to_last_time_seen[name] + min_elapsed_to_play < cur_time):
            play_sound_for_name(name)
        name_to_last_time_seen[name] = cur_time


def demo(det_models,
         face_id_model,
         reference,
         crop_size,
         id_npy,
         max_size,
         font):
    cap = cv2.VideoCapture(0)
    name_to_last_time_seen = {}

    try:
        while cap.isOpened():
            start_time = time.time()
            ret, image_np = cap.read()
            if ret and cap.isOpened():
                # Process frame
                # BGR -> RGB
                pil_img = Image.fromarray(image_np[..., ::-1])
                pil_img.thumbnail((max_size, max_size))
                viz_img, names = process_and_viz_img(
                    pil_img=pil_img,
                    det_models=det_models,
                    face_id_model=face_id_model,
                    reference=reference,
                    crop_size=crop_size,
                    id_npy=id_npy,
                    font=font,
                )
                cur_time = time.time()
                play_sound_if_needed(
                    names=names,
                    name_to_last_time_seen=name_to_last_time_seen,
                    cur_time=cur_time)

                fps = 1.0 / (cur_time - start_time)
                draw_fps(
                    img=viz_img,
                    font=font,
                    fps=fps,
                )
                # Display the resulting frame
                viz_img_bgr = np.array(viz_img)[..., ::-1]
                cv2.imshow('Face Detection Demo', viz_img_bgr)
            # Quit if we press 'q'.
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        # When everything is done, release the capture.
        cap.release()
        cv2.destroyAllWindows()


def process_files(input_dir,
                  output_dir,
                  det_models,
                  face_id_model,
                  reference,
                  crop_size,
                  id_npy,
                  max_size,
                  font):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_names = list(os.listdir(input_dir))
    for img_idx in tqdm(range(len(image_names))):
        image_name = image_names[img_idx]
        pil_img = Image.open(os.path.join(input_dir, image_name))
        pil_img.thumbnail((max_size, max_size))
        viz_img, _ = process_and_viz_img(
            pil_img=pil_img,
            det_models=det_models,
            face_id_model=face_id_model,
            reference=reference,
            crop_size=crop_size,
            id_npy=id_npy,
            font=font,
        )
        viz_img.save(os.path.join(output_dir, '{}-stream.jpg'.format(img_idx)))


def main(mode, face_id_model_root, id_features_dir, font_path):
    print('Loading models...')
    det_models = load_detect_faces_models()
    face_id_model = load_face_id_model(model_root=face_id_model_root)
    id_npy = load_id_files(id_features_dir)
    crop_size = 112
    max_size = 1024
    reference = get_reference_facial_points(default_square=True)
    font = ImageFont.FreeTypeFont(font=font_path, size=24)

    print('Starting image processing...')
    if mode == Mode.DEMO:
        demo(
            det_models=det_models,
            face_id_model=face_id_model,
            reference=reference,
            crop_size=crop_size,
            id_npy=id_npy,
            max_size=max_size,
            font=font)
    elif mode == Mode.FILE:
        process_files(
            input_dir=STREAM_DIR,
            output_dir=RESULT_DIR,
            det_models=det_models,
            face_id_model=face_id_model,
            reference=reference,
            crop_size=crop_size,
            id_npy=id_npy,
            max_size=max_size,
            font=font)
    else:
        raise ValueError('Invalid mode: {}'.format(mode))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=Mode.from_string, default=Mode.DEMO, choices=list(Mode))
    parser.add_argument('--face_id_model_root',
                        type=str,
                        default=FACE_ID_MODEL_ROOT)
    parser.add_argument('--id_features_dir',
                        type=str,
                        default=ID_FEATURES_DIR)
    parser.add_argument('--font_path',
                        type=str,
                        default=FONT_PATH)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.mode,
         args.face_id_model_root,
         args.id_features_dir,
         args.font_path)
