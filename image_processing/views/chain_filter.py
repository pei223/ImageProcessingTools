from django.shortcuts import render
from django.views import View

from image_processing.views import get_filter_and_error_message, serialized_filter_params
from ..forms import UploadFileForm
from ..models import *
from ..views import filter_page
import datetime
from django.conf import settings


class ChainFilterView(View):
    def get(self, request):
        context = filter_page.base_context(True)
        context.update({
            'form': UploadFileForm(),
            'is_showing_images': False,
        })
        delete_old_files(settings.MEDIA_ROOT,
                         datetime.datetime.now() - datetime.timedelta(minutes=settings.IMAGE_SAVING_TERMS_MIN))
        return render(request, 'filter_page.html', context)

    def post(self, request):
        form = UploadFileForm(request.POST, request.FILES)
        if not form.is_valid():
            pass

        file = request.FILES['file']

        filter_list, error_message_list = [NoFilter()], []
        if not 0 <= len(request.POST.getlist('filter-select', [])) <= settings.FILTER_COUNT_THRESHOLD:
            error_message_list.append("フィルタ数の上限は{}です.".format(str(settings.FILTER_COUNT_THRESHOLD)))
        else:
            for i, filter_type in enumerate(request.POST.getlist('filter-select', None)):
                filter_obj, error_message = get_filter_and_error_message(filter_type, request.POST, i, is_batch=False)
                if error_message != "":
                    error_message_list.append(error_message)
                if not filter_obj is None:
                    filter_list.append(filter_obj)

        error_message_of_filter_order = get_error_message_of_filter_order(filter_list)
        if error_message_of_filter_order != "":
            error_message_list.append(error_message_of_filter_order)

        serialized_form_params = serialized_filter_params(request.POST)
        if len(error_message_list) > 0:
            context = filter_page.base_context(True)
            context.update({
                "is_showing_images": False,
                "error_message_list": error_message_list,
                'form': form,
                'filter_param_list': serialized_form_params
            })
            return render(request, 'filter_page.html', context)

        origin_file_path = generate_absolute_filepath(file.name)
        save_uploaded_file(origin_file_path, file)

        origin_img = load_image(origin_file_path)

        extension = get_extension(file.name)
        origin_filename_excluded_extension = get_file_name_excluded_extension(file.name)

        filtered_images = []
        code_generator = FilterCodeGenerator()
        current_img = origin_img
        for filter_type in filter_list:
            filtered_filename = filter_type.get_arranged_filename(origin_filename_excluded_extension)
            filtered_filepath = generate_absolute_filepath(filtered_filename + extension)
            filtered_img = filter_type.filtering(current_img)
            save_image(filtered_img, filtered_filepath)
            filtered_images.append(
                FilteredImage(generate_filepath_for_display(filtered_filename + extension), filter_type))
            current_img = filtered_img
            code_generator.add(filter_type.generate_code())

        context = filter_page.base_context(True)
        context.update({
            "filtered_images": filtered_images,
            "is_showing_images": True,
            "generated_code": code_generator.generate(),
            "filter_count": code_generator.arranged_row_count(),
            'form': form,
            'filter_param_list': serialized_form_params
        })

        return render(request, 'filter_page.html', context)
