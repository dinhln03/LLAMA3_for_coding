import argparse
import sys
from os import path

import cv2 as cv

from mcrops import veget, utils


def full_imshow(name, image):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.resizeWindow(name, 800, 600)
    cv.imshow(name, image)


def main(image_path: str, resolution: float, row_sep: float):
    print(f'Starting analysis.\n')
    print(f'Loading image {image_path}')
    # Load a crop field image
    image = cv.imread(image_path)

    if image is None:
        raise ValueError(f'Unable to load image {image_path}')

    h, w = image.shape[:2]
    image_draw = image.copy()
    print(f'Image loaded. Size is {w}x{h} pixels.')

    print('Segmenting vegetation')
    # Segment vegetation
    veg_mask = veget.segment_vegetation(image)

    print('Detecting crop area')
    # Detect the crop field ROI area
    roi_poly = veget.detect_roi(
        veg_mask, row_sep=row_sep, resolution=resolution
    )
    # Draw the contours of the ROI area
    cv.drawContours(
        image=image_draw,
        contours=[roi_poly],
        contourIdx=-1,
        color=(0, 0, 255),
        thickness=8,
        lineType=cv.LINE_AA
    )
    # Build a mask image from the ROI polyline
    roi_mask = utils.poly_mask(roi_poly, veg_mask.shape[:2])
    veg_mask[roi_mask == 0] = 0

    print('Computing vegetation density map')
    # Create a vegetation density map from the vegetation mask
    density_map = veget.mask_density(
        veg_mask,
        roi_mask,
        resolution=resolution,
        cell_size=(8, 8)
    )
    d_min = density_map.min()
    d_max = density_map.max()
    print(f'Vegetation density is range [{d_min:.3f}, {d_max:.3f}]')

    # Convert the vegetation density map to a color image
    image_map = utils.array_image(density_map, colormap=cv.COLORMAP_JET)

    full_imshow('Crop field image', image_draw)
    full_imshow('Vegetation mask', veg_mask)
    full_imshow('Vegetation density map', image_map)

    print(f'Analysis finished. Press any key to quit.\n')
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':

    curr_dir = path.dirname(path.abspath(__file__))
    parent_dir, _ = path.split(curr_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        type=str,
        required=False,
        default=path.join(curr_dir, 'data/crop_field_sparse_res40.png'),
        help='Path to crop field image.'
    )
    parser.add_argument(
        '--res',
        type=float,
        required=False,
        default=40,
        help='Image resolution in pixels/meter.'
    )
    parser.add_argument(
        '--row_sep',
        type=float,
        required=False,
        default=0.7,
        help='Approximated mean crop rows separation in meters.'
    )
    args = parser.parse_args(sys.argv[1:])

    main(image_path=args.input, resolution=args.res, row_sep=args.row_sep)
