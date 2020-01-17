from ..models import *


def base_context(is_chain):
    return {
        "is_chain": is_chain,
        "filter_name_list": filter_name_list,
        "rule_dict": rule_dict,
        "sobel_kind_list": zip(SobelFilter.kind_names, SobelFilter.kind_values),
    }
