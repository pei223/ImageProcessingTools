from django.views.generic import RedirectView


class RootPageView(RedirectView):
    url = "/image_processing/batch_filter"
