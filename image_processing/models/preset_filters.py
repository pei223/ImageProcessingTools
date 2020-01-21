from ..models import *


class PresetFilters(metaclass=ABCMeta):
    @abstractmethod
    def preset_filters(self) -> List[Dict]:
        pass

    @abstractmethod
    def filters_id(self) -> int:
        pass

    @abstractmethod
    def filters_name(self) -> str:
        pass


class DamageDetectPresetFilters(PresetFilters):
    def filters_name(self):
        return "キズ検出向きフィルターセット"

    def filters_id(self):
        return 1

    def preset_filters(self):
        filter_len = 6

        filter_list: List[Dict] = []
        for i in range(filter_len):
            filter_list.append(default_form_value())

        filter_value = {
            "filter_select": str(filter_name_list.index(GrayScaleFilter.filter_name()))
        }
        filter_list[0].update(filter_value)

        filter_value = {
            "filter_select": str(filter_name_list.index(SharpFilter.filter_name())),
            "sharp_k": "3",
        }
        filter_list[1].update(filter_value)

        filter_value = {
            "filter_select": str(filter_name_list.index(GaussianFilter.filter_name())),
            "gauss_kernel_size": "5",
        }
        filter_list[2].update(filter_value)

        filter_value = {
            "filter_select": str(filter_name_list.index(SobelFilter.filter_name())),
        }
        filter_list[3].update(filter_value)

        filter_value = {
            "filter_select": str(filter_name_list.index(BinaryOtsuFilter.filter_name())),
        }
        filter_list[4].update(filter_value)

        filter_value = {
            "filter_select": str(filter_name_list.index(ClosingFilter.filter_name())),
        }
        filter_list[5].update(filter_value)

        return filter_list


preset_filters = [
    DamageDetectPresetFilters(),
]


def get_preset_filters_id_or_none(id_s: str) -> PresetFilters or None:
    if not id_s.isdigit():
        return None
    id_i = int(id_s)
    for filters in preset_filters:
        if id_i == filters.filters_id():
            return filters
    return None
