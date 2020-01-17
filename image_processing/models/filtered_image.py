from ..models import BaseFilter


class FilteredImage:
    def __init__(self, image_url, filter_type: BaseFilter):
        self.image_url = image_url
        self.filter = filter_type
