from django import forms
from .models import Dataset

class DatasetForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['dataset_file']


from django import forms

class TestDataForm(forms.Form):
    review_text = forms.CharField(
        label="Enter Review Text",
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3})
    )
