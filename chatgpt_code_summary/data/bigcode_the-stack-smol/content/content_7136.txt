import cv2

from util.image_type import ColorImage


class VideoWriter:
    """
    This class wraps a cv2.VideoWriter object,
    preset some parameters so simpler to use.
    """

    def __init__(self, video_path: str, fps: float = 10.0) -> None:
        """
        Arguments:
            video_path: The path to output the video.
            fps: If higher than the writing rate, the video will be fast-forwarded.
        """
        fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")
        self._video_writer = cv2.VideoWriter(video_path + ".mp4", fourcc, fps, (640, 480))

    def write(self, image: ColorImage) -> None:
        """Writes the next video frame."""
        self._video_writer.write(image)

    def is_opened(self) -> bool:
        """Returns True if video writer has been successfully initialized."""
        return self._video_writer.isOpend()

    def release(self) -> None:
        """Closes the video writer."""
        self._video_writer.release()
