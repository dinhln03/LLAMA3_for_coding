import numpy
import os

from src.readers import SimpleBSONReader


def read_train_example(path='../../data/train_example.bson', pca_reduction=True):
    read_result = SimpleBSONReader.read_all(path)
    pixels = read_result.pixel_matrix
    numpy.savetxt("../../out/train_example.csv", pixels, delimiter=",", fmt='%.d')
    if pca_reduction:
        pixel_reduced = read_result.pixel_matrix_reduced
        numpy.savetxt("../../out/pca_train_example.csv", pixel_reduced, delimiter=",", fmt='%.d')
    return pixels


def read_and_save_intermediate(path='../../data/train_example.bson', pca_reduction=True,
                               file_out_path="../../out/train_example.csv",
                               reduced_file_out_path="../../out/pca_train_example.csv", root_path="../../out/",
                               n_components=90, first_occurence_number=1):

    dirname = os.path.dirname(file_out_path)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    read_result = SimpleBSONReader.read_all(
        path,
        save_intermediate=True,
        save_png=True,
        root_path=root_path,
        first_occurence_number=first_occurence_number,
        n_components=n_components)
    pixels = read_result.pixel_matrix
    numpy.savetxt(file_out_path, pixels, delimiter=",", fmt='%.d')
    if pca_reduction:
        dirname = os.path.dirname(reduced_file_out_path)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        pixel_reduced = read_result.pixel_matrix_reduced
        numpy.savetxt(reduced_file_out_path, pixel_reduced, delimiter=",", fmt='%s')
    return pixels


if __name__ == "__main__":
    read_and_save_intermediate()
