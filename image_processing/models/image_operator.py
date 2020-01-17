import numpy as np
import os
import cv2
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from datetime import datetime


def save_image(img: np.ndarray, file_path: str):
    cv2.imwrite(file_path, img)


def load_image(file_path: str):
    return cv2.imread(file_path)


def get_file_name_excluded_extension(file_path: str) -> str:
    return file_path[:file_path.index(".")]


def get_extension(file_path: str) -> str:
    return file_path[file_path.index("."):]


def generate_absolute_filepath(file_name: str):
    return "{}/{}".format(settings.MEDIA_ROOT, file_name)


def save_uploaded_file(file_path: str, file):
    FileSystemStorage().save(file_path, file)


def generate_filepath_for_display(file_name: str) -> str:
    return "{}/{}".format(settings.MEDIA_URL, file_name)


def delete_old_files(dir_path: str, threshold_datetime: datetime):
    for filename in os.listdir(dir_path):
        file_path = '{}/{}'.format(dir_path, filename)
        stat = os.stat(file_path)
        if not stat:
            continue
        timestamp = datetime.fromtimestamp(stat.st_mtime)
        if timestamp.time() < threshold_datetime.time():
            os.remove(file_path)
