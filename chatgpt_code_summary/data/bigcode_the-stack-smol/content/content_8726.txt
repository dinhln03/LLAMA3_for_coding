from pathlib import Path, PurePath


class Directory:

    def __init__(self, path_in_image, path_out_image, path_out_file):
        self.__path_in_image = path_in_image
        self.__path_out_image = path_out_image
        self.__path_out_file = path_out_file
        self.__allowed_image_extension = ['.jpg', '.jpeg', '.png']

    @property
    def get_path_in_image(self):
        return self.__path_in_image

    @property
    def get_path_out_image(self):
        return self.__path_out_image

    @property
    def get_path_out_file(self):
        return self.__path_out_file

    @property
    def get_allowed_image_extension(self):
        return self.__allowed_image_extension

    def get_images(self):
        file_path_list = []

        for file in self.get_path_in_image.iterdir():
            if file.is_file():
                for image_extension in self.get_allowed_image_extension:
                    if file.suffix == image_extension:
                        file_path_list.append(PurePath.joinpath(
                            self.get_path_in_image, file))
        return file_path_list

    def move_decoded_image(self, path, operation_id):
        file_extension = path.suffix
        new_file_name = operation_id + file_extension
        Path.rename(path, self.get_path_out_image / new_file_name)
