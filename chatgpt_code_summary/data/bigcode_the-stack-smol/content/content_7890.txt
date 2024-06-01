from django import forms
from django.forms import ModelForm

from .models import Review


class ReviewForm(ModelForm):
    required_css_class = 'required'

    def __init__(self, *args, **kwargs):
        """
        user object is passed to the form in kwargs in the view
        the user objected is removed from kwargs and then the
        super class form object is instantiated. This is because
        our form needs the user object not its super class.
        """
        self.user = kwargs.pop('user', None)
        super(ReviewForm, self).__init__(*args, **kwargs)

    class Meta:
        model = Review
        fields = [
            'title', 'review', 'type', 'book'
        ]

    def clean_book(self, *args, **kwargs):
        """
        This method checks if a user has already reviewed
        the selected book. As per django docs exists() is
        an efficient way of checking this.
        """
        book = self.cleaned_data.get("book")
        if Review.objects.filter(book=book, author=self.user).exists():
            raise forms.ValidationError("Book already reviewed by user {}".format(self.user))
        else:
            return book
