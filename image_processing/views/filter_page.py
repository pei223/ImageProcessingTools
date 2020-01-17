from typing import Tuple
from ..models import *


def base_context(is_chain):
    return {
        "is_chain": is_chain,
        "filter_name_list": filter_name_list,
        "rule_dict": rule_dict,
        "sobel_kind_list": zip(SobelFilter.kind_names, SobelFilter.kind_values),
    }


def get_filter_and_error_message(filter_type: str, post, i: int, is_batch: bool = True) -> Tuple[
    BaseFilter or None, str]:
    if filter_type == "0":
        error_message = GaussianFilter.validation(post.getlist("gauss-kernel-size")[i], post.getlist("gauss-sigma")[i])
        if error_message != "":
            return None, error_message
        return GaussianFilter(int(post.getlist("gauss-kernel-size")[i]),
                              int(post.getlist("gauss-sigma")[i])), ""
    elif filter_type == "1":
        error_message = MedianFilter.validation(post.getlist("median-kernel-size")[i])
        if error_message != "":
            return None, error_message
        return MedianFilter(int(post.getlist("median-kernel-size")[i])), ""

    elif filter_type == "2":
        error_message = SharpFilter.validation(post.getlist("sharp-k")[i])
        if error_message != "":
            return None, error_message
        return SharpFilter(int(post.getlist("sharp-k")[i])), ""

    elif filter_type == "3":
        error_message = SobelFilter.validation(post.getlist("sobel-kernel-size")[i],
                                               post.getlist("sobel-kind")[i])
        if error_message != "":
            return None, error_message
        return SobelFilter(int(post.getlist("gauss-kernel-size")[i]), int(post.getlist("sobel-kind")[i])), ""

    elif filter_type == "4":
        error_message = LaplacianFilter.validation(post.getlist("laplacian-kernel-size")[i])
        if error_message != "":
            return None, error_message
        return LaplacianFilter(int(post.getlist("laplacian-kernel-size")[i])), ""

    elif filter_type == "5":
        return GrayScaleFilter(), ""

    elif filter_type == "6":
        return BinaryOtsuFilter(is_batch), ""

    elif filter_type == "7":
        error_message = BinaryThresholdFilter.validation(post.getlist("binary-threshold-lower")[i],
                                                         post.getlist("binary-threshold-upper")[i])
        if error_message != "":
            return None, error_message
        return BinaryThresholdFilter(int(post.getlist("binary-threshold-lower")[i]),
                                     int(post.getlist("binary-threshold-upper")[i]),
                                     is_batch), ""

    elif filter_type == "8":
        error_message = ClosingFilter.validation(post.getlist("closing-kernel-size")[i])
        if error_message != "":
            return None, error_message
        return ClosingFilter(int(post.getlist("closing-kernel-size")[i]), is_batch), ""

    elif filter_type == "9":
        error_message = OpeningFilter.validation(post.getlist("opening-kernel-size")[i])
        if error_message != "":
            return None, error_message
        return OpeningFilter(int(post.getlist("opening-kernel-size")[i]), is_batch), ""

    elif filter_type == "10":
        error_message = BilateralFilter.validation(post.getlist("bilateral-kernel-size")[i],
                                                   post.getlist("bilateral-sigma-color")[i],
                                                   post.getlist("bilateral-sigma-space")[i])
        if error_message != "":
            return None, error_message
        return BilateralFilter(int(post.getlist("bilateral-kernel-size")[i]),
                               float(post.getlist("bilateral-sigma-color")[i]),
                               float(post.getlist("bilateral-sigma-space")[i])), ""
    elif filter_type == "11":
        error_message = AreaThresholdFilter.validation(post.getlist('area-threshold-lower')[i],
                                                       post.getlist('area-threshold-upper')[i])
        if error_message != "":
            return None, error_message
        return AreaThresholdFilter(int(post.getlist('area-threshold-lower')[i]),
                                   int(post.getlist('area-threshold-upper')[i])), ""
    else:
        return None, ""


def serialized_filter_params(post):
    param_list = []
    for i in range(len(post.getlist('filter-select'))):
        param_list.append({
            "filter_select": post.getlist('filter-select')[i],
            "gauss_kernel_size": post.getlist('gauss-kernel-size')[i],
            "gauss_sigma": post.getlist('gauss-sigma')[i],
            "median_kernel_size": post.getlist('median-kernel-size')[i],
            "sharp_k": post.getlist('sharp-k')[i],
            "sobel_kernel_size": post.getlist('sobel-kernel-size')[i],
            "sobel_kind": post.getlist('sobel-kind')[i],
            "laplacian_kernel_size": post.getlist('laplacian-kernel-size')[i],
            "binary_threshold_lower": post.getlist('binary-threshold-lower')[i],
            "binary_threshold_upper": post.getlist('binary-threshold-upper')[i],
            "closing_kernel_size": post.getlist('closing-kernel-size')[i],
            "opening_kernel_size": post.getlist('opening-kernel-size')[i],
            "bilateral_kernel_size": post.getlist('bilateral-kernel-size')[i],
            "bilateral_sigma_color": post.getlist('bilateral-sigma-color')[i],
            "bilateral_sigma_space": post.getlist('bilateral-sigma-space')[i],
            "area_threshold_lower": post.getlist('area-threshold-lower')[i],
            "area_threshold_upper": post.getlist('area-threshold-upper')[i],
        })
    return param_list
