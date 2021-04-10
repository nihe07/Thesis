from django import forms 
from .models import Image
import os
  
class ImageForm(forms.ModelForm):
    # """Form for the image model"""
    class Meta:
        model = Image
        fields = ('image',)
        widgets = {
            'image': forms.FileInput(attrs={'id': 'upload-im'}),
        }