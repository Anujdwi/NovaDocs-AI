# retrieval/forms.py
from django import forms

class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=255, required=True)
    file = forms.FileField(required=True)
