from django.urls import path, include
from django.contrib import admin

from django.conf import settings # new
from django.conf.urls.static import static # new


import hello.views


urlpatterns = [
    path("admin/", admin.site.urls),
    path("", hello.views.image_upload_view, name="image_upload_view"),
    path("return_palinim/", hello.views.return_palinim, name="return_palinim"),
    path("return_hue/", hello.views.return_hue, name="return_hue"),
    path("return_pix/", hello.views.return_pix, name="return_pix"),
    path("returnAll/", hello.views.returnAll, name="returnAll"),
    path("returnMET/", hello.views.returnMET, name="returnMET"),
    path("returnMETvals/", hello.views.returnMETvals, name="returnMET"),
    path("returnWordsearch/", hello.views.returnWordsearch, name="returnWordsearch")
]

if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)
