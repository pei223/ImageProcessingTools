from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np
import cv2


class BaseFilter(metaclass=ABCMeta):
    @abstractmethod
    def filter_name(self) -> str:
        pass

    @abstractmethod
    def filtering(self, img: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def args_description(self) -> str:
        pass

    @abstractmethod
    def generate_code(self) -> str:
        pass

    @abstractmethod
    def get_arranged_filename(self, file_name_excluded_extension) -> str:
        pass

    def required_gray_scale_in_advance(self):
        return False


class NoFilter(BaseFilter):
    def filter_name(self):
        return "元の画像"

    def filtering(self, img: np.ndarray):
        return img

    def args_description(self):
        return ""

    def generate_code(self):
        return ""

    def get_arranged_filename(self, file_name_excluded_extension):
        return file_name_excluded_extension


class GaussianFilter(BaseFilter):
    def __init__(self, kernel_size: int, sigma: float):
        self._kernel_size = kernel_size
        self._sigma = sigma

    def filter_name(self):
        return "ガウシアンフィルタ"

    def filtering(self, img: np.ndarray):
        return cv2.GaussianBlur(img, (self._kernel_size, self._kernel_size), self._sigma)

    def args_description(self):
        return "カーネルサイズ: {},  分散: {}".format(self._kernel_size, self._sigma)

    def generate_code(self):
        return "# {}\nimg = cv2.GaussianBlur(img, ({}, {}), {})".format(self.filter_name(), str(self._kernel_size),
                                                                        str(self._kernel_size),
                                                                        str(self._sigma))

    def get_arranged_filename(self, file_name_excluded_extension):
        return file_name_excluded_extension + "_gauss"

    @staticmethod
    def validation(kernel_size_s: str, sigma_s: str) -> str:
        result = ["[ガウシアンフィルタ] ", ]
        try:
            kernel_size = int(kernel_size_s)
            if kernel_size % 2 == 0:
                raise RuntimeError
            if not rule_dict["kernel_min"] <= kernel_size < rule_dict["kernel_max"]:
                raise RuntimeError
        except:
            result.append("カーネルサイズは0〜49の奇数にしてください. ")

        try:
            sigma = float(sigma_s)
            if not 0.0 <= sigma <= 100.0:
                raise RuntimeError
        except:
            result.append("ガウシアンフィルタの分散は0〜100の小数にしてください. ")
        if len(result) != 1:
            return "".join(result)
        return ""


class MedianFilter(BaseFilter):
    def __init__(self, kernel_size: int):
        self._kernel_size = kernel_size

    def filter_name(self):
        return "メディアンフィルタ"

    def filtering(self, img: np.ndarray):
        return cv2.medianBlur(img, self._kernel_size)

    def args_description(self):
        return "カーネルサイズ: {}".format(self._kernel_size)

    def generate_code(self):
        return "# {}\nimg = cv2.medianBlur(img, {})".format(self.filter_name(), str(self._kernel_size))

    def get_arranged_filename(self, file_name_excluded_extension):
        return file_name_excluded_extension + "_median"

    @staticmethod
    def validation(kernel_size_s: str) -> str:
        try:
            kernel_size = int(kernel_size_s)
            if kernel_size % 2 == 0:
                raise RuntimeError
            if not rule_dict["kernel_min"] <= kernel_size < rule_dict["kernel_max"]:
                raise RuntimeError
            return ""
        except:
            return "[メディアンフィルタ] カーネルサイズは0〜49の奇数である必要があります. "


class SharpFilter(BaseFilter):
    def __init__(self, k: int):
        self._k = k

    def filter_name(self):
        return "鮮鋭化"

    def filtering(self, img: np.ndarray):
        kernel = np.array(
            [[-self._k / 9, -self._k / 9, -self._k / 9], [-self._k / 9, 1 + 8 * self._k / 9, -self._k / 9],
             [-self._k / 9, -self._k / 9, -self._k / 9]], np.float32)
        return cv2.filter2D(img, -1, kernel)

    def args_description(self):
        return "k: {}".format(self._k)

    def generate_code(self):
        return """# {comment}
kernel = np.array(
    [[-{k} / 9, -{k} / 9, -{k} / 9], [-{k} / 9, 1 + 8 * {k} / 9, -{k} / 9],
     [-{k} / 9, -{k} / 9, -{k} / 9]], np.float32)
img = cv2.filter2D(img, -1, kernel)""".format(comment=self.filter_name(), k=str(self._k))

    def get_arranged_filename(self, file_name_excluded_extension):
        return file_name_excluded_extension + "_sharp"

    @staticmethod
    def validation(k_s: str) -> str:
        try:
            k = int(k_s)
            if not 1 <= k <= 100:
                raise RuntimeError
            return ""
        except:
            return "[鮮鋭化] kは1〜100の間の値にしてください. "


class SobelFilter(BaseFilter):
    kind_values = [
        0, 1, 2
    ]

    kind_names = [
        "縦横平均", "縦", "横",
    ]

    def __init__(self, kernel_size: int, kind: int):
        self._kernel_size = kernel_size
        self._kind = kind

    def filter_name(self):
        return "Sobelフィルタ"

    def filtering(self, img: np.ndarray):
        if self._kind == 0:
            return ((abs(cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=self._kernel_size)) + \
                     abs(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=self._kernel_size))) // 2).astype('uint8')
        elif self._kind == 1:
            return cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=self._kernel_size)
        elif self._kind == 2:
            return cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=self._kernel_size)
        else:
            assert "不適切なkind"

    def args_description(self):
        return "カーネルサイズ: {}, 種別: {}".format(self._kernel_size, SobelFilter.kind_names[self._kind])

    def generate_code(self):
        if self._kind == 0:
            return "# {comment}\nimg = ((abs(cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize={kernel_size})) + abs(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize={kernel_size}))) // 2).astype('uint8')".format(
                comment=self.filter_name(),
                kernel_size=str(self._kernel_size))
        elif self._kind == 1:
            return "# {}\nimg = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize={})".format(self.filter_name(),
                                                                                   str(self._kernel_size))
        elif self._kind == 2:
            return "# {}\nimg = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize={})".format(self.filter_name(),
                                                                                   str(self._kernel_size))

        else:
            assert "不適切なkind"

    def get_arranged_filename(self, file_name_excluded_extension):
        return file_name_excluded_extension + "_sobel"

    @staticmethod
    def validation(kernel_size_s: str, kind_s: str) -> str:
        result = ["[Sobelフィルタ] ", ]
        try:
            kernel_size = int(kernel_size_s)
            if kernel_size % 2 == 0:
                raise RuntimeError
            if not rule_dict["kernel_min"] <= kernel_size < rule_dict["kernel_max"]:
                raise RuntimeError
        except:
            result.append("カーネルサイズは0〜49の奇数にしてください. ")

        try:
            kind = int(kind_s)
            if not min(SobelFilter.kind_values) <= kind <= max(SobelFilter.kind_values):
                raise RuntimeError
        except:
            # 通常通らない
            result.append("種別が不適切です. ")
        if len(result) != 1:
            return "".join(result)
        return ""


class LaplacianFilter(BaseFilter):
    def __init__(self, kernel_size: int):
        self._kernel_size = kernel_size

    def filter_name(self):
        return "ラプラシアンフィルタ"

    def filtering(self, img: np.ndarray):
        return cv2.Laplacian(img, cv2.CV_32F, ksize=self._kernel_size)

    def args_description(self):
        return "カーネルサイズ: {}".format(self._kernel_size)

    def generate_code(self):
        return "# {}\nimg = cv2.Laplacian(img, cv2.CV_32F, ksize={})".format(self.filter_name(), str(self._kernel_size))

    def get_arranged_filename(self, file_name_excluded_extension):
        return file_name_excluded_extension + "_laplacian"

    @staticmethod
    def validation(kernel_size_s: str) -> str:
        try:
            kernel_size = int(kernel_size_s)
            if kernel_size % 2 == 0:
                raise RuntimeError
            if not rule_dict["kernel_min"] <= kernel_size < rule_dict["kernel_max"]:
                raise RuntimeError
            return ""
        except:
            return "[ラプラシアンフィルタ] カーネルサイズは0〜49の奇数である必要があります. "


class GrayScaleFilter(BaseFilter):
    def filter_name(self):
        return "グレースケール"

    def filtering(self, img: np.ndarray):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def args_description(self):
        return ""

    def generate_code(self):
        return "# {}\nimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)".format(self.filter_name())

    def get_arranged_filename(self, file_name_excluded_extension):
        return file_name_excluded_extension + "_gray"


class BinaryOtsuFilter(BaseFilter):
    def __init__(self, required_gray_scale=True):
        self._required_gray_scale = required_gray_scale
        self._threshold = -1

    def filter_name(self):
        return "大津の2値化"

    def filtering(self, img: np.ndarray):
        if not self._required_gray_scale:
            threshold, result_img = cv2.threshold(img.astype('uint8'), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self._threshold = threshold
            return result_img

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold, result_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self._threshold = threshold
        return result_img

    def args_description(self):
        if self._threshold == -1:
            return ""
        return "閾値: {}".format(str(self._threshold))

    def generate_code(self):
        if not self._required_gray_scale:
            return "# {}\nimg = cv2.threshold(img.astype('uint8'), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]".format(
                self.filter_name())
        return "# {}\ngray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nimg = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]".format(
            self.filter_name())

    def get_arranged_filename(self, file_name_excluded_extension):
        return file_name_excluded_extension + "_otsu"

    def required_gray_scale_in_advance(self):
        return True


class BinaryThresholdFilter(BaseFilter):
    def __init__(self, threshold_lower: int, threshold_upper: int, required_gray_scale=True):
        self._threshold_lower, self._threshold_upper = threshold_lower, threshold_upper
        self._required_gray_scale = required_gray_scale

    def filter_name(self):
        return "閾値による2値化"

    def filtering(self, img: np.ndarray):
        if not self._required_gray_scale:
            return cv2.inRange(img, self._threshold_lower, self._threshold_upper)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.inRange(gray_img, self._threshold_lower, self._threshold_upper)

    def args_description(self):
        return "閾値: {}〜{}".format(self._threshold_lower, self._threshold_upper)

    def generate_code(self):
        if not self._required_gray_scale:
            return "# {}\nimg = cv2.inRange(gray_img, {}, {})".format(
                self.filter_name(), str(self._threshold_lower), str(self._threshold_upper))
        return "# {}\ngray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nimg = cv2.inRange(gray_img, {}, {})".format(
            self.filter_name(), str(self._threshold_lower), str(self._threshold_upper))

    def get_arranged_filename(self, file_name_excluded_extension):
        return file_name_excluded_extension + "_binary"

    @staticmethod
    def validation(threshold_lower_s: str, threshold_upper_s: str) -> str:
        result = ["[閾値による2値化] ", ]
        try:
            threshold_lower = int(threshold_lower_s)
            threshold_upper = int(threshold_upper_s)
            if not 0 <= threshold_lower <= 255 or not 0 <= threshold_upper <= 255:
                raise RuntimeError
            if threshold_lower > threshold_upper:
                result.append("閾値は上限の方が下限より大きい必要があります. ")
        except:
            result.append("閾値は0〜255である必要があります. ")
        if len(result) != 1:
            return "".join(result)
        return ""

    def required_gray_scale_in_advance(self):
        return True


class ClosingFilter(BaseFilter):
    def __init__(self, closing_kernel_size: int, required_gray_scale=True):
        self._kernel_size = closing_kernel_size
        self._required_gray_scale = required_gray_scale

    def filter_name(self) -> str:
        return "クロージング"

    def filtering(self, img: np.ndarray) -> np.ndarray:
        if not self._required_gray_scale:
            return cv2.morphologyEx(img.astype('uint8'), cv2.MORPH_CLOSE,
                                    np.ones((self._kernel_size, self._kernel_size), np.uint8))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, np.ones((self._kernel_size, self._kernel_size), np.uint8))

    def args_description(self) -> str:
        return "カーネルサイズ: {}".format(str(self._kernel_size))

    def generate_code(self) -> str:
        if not self._required_gray_scale:
            return "# {comment}\nimg = cv2.morphologyEx(img.astype('uint8'), cv2.MORPH_CLOSE, np.ones(({kernel_size}, {kernel_size}), np.uint8))".format(
                comment=self.filter_name(), kernel_size=str(self._kernel_size))
        return "# {comment}\ngray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nimg = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, np.ones(({kernel_size}, {kernel_size}), np.uint8))".format(
            comment=self.filter_name(), kernel_size=str(self._kernel_size))

    def get_arranged_filename(self, file_name_excluded_extension) -> str:
        return file_name_excluded_extension + "_closing"

    @staticmethod
    def validation(kernel_size_s: str) -> str:
        try:
            kernel_size = int(kernel_size_s)
            if not rule_dict["kernel_min"] <= kernel_size < rule_dict["kernel_max"]:
                raise RuntimeError
            return ""
        except:
            return "[クロージング] カーネルサイズは0〜49である必要があります. "


class OpeningFilter(BaseFilter):
    def __init__(self, kernel_size: int, required_gray_scale=True):
        self._kernel_size = kernel_size
        self._required_gray_scale = required_gray_scale

    def filter_name(self) -> str:
        return "オープニング"

    def filtering(self, img: np.ndarray) -> np.ndarray:
        if not self._required_gray_scale:
            return cv2.morphologyEx(img.astype('uint8'), cv2.MORPH_OPEN,
                                    np.ones((self._kernel_size, self._kernel_size), np.uint8))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, np.ones((self._kernel_size, self._kernel_size), np.uint8))

    def args_description(self) -> str:
        return "カーネルサイズ: {}".format(str(self._kernel_size))

    def generate_code(self) -> str:
        if not self._required_gray_scale:
            return "# {comment}\nimg = cv2.morphologyEx(img.astype('uint8'), cv2.MORPH_OPEN, np.ones(({kernel_size}, {kernel_size}), np.uint8))".format(
                comment=self.filter_name(), kernel_size=str(self._kernel_size))
        return "# {comment}\ngray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\nimg = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, np.ones(({kernel_size}, {kernel_size}), np.uint8))".format(
            comment=self.filter_name(), kernel_size=str(self._kernel_size))

    def get_arranged_filename(self, file_name_excluded_extension) -> str:
        return file_name_excluded_extension + "_opening"

    @staticmethod
    def validation(kernel_size_s: str) -> str:
        try:
            kernel_size = int(kernel_size_s)
            if not rule_dict["kernel_min"] <= kernel_size < rule_dict["kernel_max"]:
                raise RuntimeError
            return ""
        except:
            return "[オープニング] カーネルサイズは0〜49である必要があります. "


class BilateralFilter(BaseFilter):
    def __init__(self, kernel_size: int, sigma_color: float, sigma_space):
        self._kernel_size = kernel_size
        self._sigma_color = sigma_color
        self._sigma_space = sigma_space

    def filter_name(self):
        return "バイラテラルフィルタ"

    def filtering(self, img: np.ndarray):
        return cv2.bilateralFilter(img, self._kernel_size, self._sigma_color, self._sigma_space)

    def args_description(self):
        return "カーネルサイズ: {},  色の標準偏差: {}, 距離の標準偏差: {}".format(self._kernel_size, str(self._sigma_color),
                                                              str(self._sigma_space))

    def generate_code(self):
        return "# {}\nimg = cv2.bilateralFilter(img, {}, {}, {})".format(self.filter_name(), str(self._kernel_size),
                                                                         str(self._sigma_color),
                                                                         str(self._sigma_space))

    def get_arranged_filename(self, file_name_excluded_extension):
        return file_name_excluded_extension + "_bilateral"

    @staticmethod
    def validation(kernel_size_s: str, sigma_color_s: str, sigma_space_s: str) -> str:
        result = ["[バイラテラルフィルタ] ", ]
        try:
            kernel_size = int(kernel_size_s)
            if kernel_size % 2 == 0:
                raise RuntimeError
            if not rule_dict["kernel_min"] <= kernel_size < rule_dict["kernel_max"]:
                raise RuntimeError
        except:
            result.append("カーネルサイズは0〜49の奇数にしてください. ")

        try:
            sigma = float(sigma_color_s)
            if not 0.0 < sigma < 100.0:
                raise RuntimeError
        except:
            result.append("色の標準偏差は0〜100の小数にしてください. ")

        try:
            sigma = float(sigma_space_s)
            if not 0.0 < sigma < 100.0:
                raise RuntimeError
        except:
            result.append("距離の標準偏差は0〜100の小数にしてください. ")
        if len(result) != 1:
            return "".join(result)
        return ""


class AreaThresholdFilter(BaseFilter):
    def __init__(self, area_threshold_lower: int, area_threshold_upper: int):
        self._area_threshold_lower = area_threshold_lower
        self._area_threshold_upper = area_threshold_upper

    def filter_name(self):
        return "面積閾値によるフィルタリング"

    def filtering(self, img: np.ndarray):
        result_img = img.copy()
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            area = cv2.contourArea(contour.astype('int32'))
            print(area)
            if not self._area_threshold_lower <= area <= self._area_threshold_upper:
                x, y, width, height = cv2.boundingRect(contour)
                result_img[y:y + height, x:x + width] = 0
        return result_img

    def args_description(self):
        return "面積の閾値: {}〜{}".format(self._area_threshold_lower, self._area_threshold_upper)

    def generate_code(self):
        return ""

    def get_arranged_filename(self, file_name_excluded_extension):
        return file_name_excluded_extension + "_area_threshold"

    @staticmethod
    def validation(threshold_lower_s: str, threshold_upper_s: str) -> str:
        try:
            threshold_lower = int(threshold_lower_s)
            if not rule_dict["area_threshold_min"] <= threshold_lower <= rule_dict["area_threshold_max"]:
                raise RuntimeError

            threshold_upper = int(threshold_upper_s)
            if not rule_dict["area_threshold_min"] <= threshold_upper <= rule_dict["area_threshold_max"]:
                raise RuntimeError
        except:
            return ("[面積閾値によるフィルタリング] 閾値は{}〜{}の整数である必要があります. ".format(rule_dict["area_threshold_min"],
                                                                      rule_dict["area_threshold_max"]))
        return ""

    def required_gray_scale_in_advance(self):
        return True


def get_error_message_of_filter_order(filter_list: List[BaseFilter]) -> str:
    is_gray_scaled = False
    for filter_obj in filter_list:
        if isinstance(filter_obj, GrayScaleFilter):
            if is_gray_scaled:
                return "[フィルターの並び順] グレースケールは二回できません. "
            is_gray_scaled = True

        if filter_obj.required_gray_scale_in_advance() and not is_gray_scaled:
            return "[フィルターの並び順] {}は事前にグレースケールする必要があります. ".format(filter_obj.filter_name())
    return ""


filter_name_list = [
    GaussianFilter(1, 0).filter_name(),
    MedianFilter(3).filter_name(),
    SharpFilter(3).filter_name(),
    SobelFilter(3, 0).filter_name(),
    LaplacianFilter(3).filter_name(),
    GrayScaleFilter().filter_name(),
    BinaryOtsuFilter().filter_name(),
    BinaryThresholdFilter(0, 255).filter_name(),
    ClosingFilter(3).filter_name(),
    OpeningFilter(3).filter_name(),
    BilateralFilter(3, 3.0, 3.0).filter_name(),
    AreaThresholdFilter(3, 3).filter_name(),
]

rule_dict = {
    "kernel_max": 49,
    "kernel_min": 3,
    "sigma_min": 0,
    "sigma_max": 100,
    "sharp_k_min": 1,
    "sharp_k_max": 100,
    "sobel_kind_min": min(SobelFilter.kind_values),
    "threshold_min": 0,
    "threshold_max": 256,
    "threshold_lower_default": 50,
    "threshold_upper_default": 255,
    "bilateral_kernel_default": 15,
    "bilateral_sigma_default": 20,
    "area_threshold_min": 0,
    "area_threshold_max": 100000000000,
    "area_threshold_lower_default": 100,
    "area_threshold_upper_default": 10000,
}
