from academicInfo.models import Department

from faculty.forms import FacultySignupForm
from faculty.models import Faculty

from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone

class FacultySignupFormTest(TestCase):

    def test_signup_form_label(self):
        form = FacultySignupForm()
        self.assertTrue(
            form.fields['first_name'].label == 'First Name' and
            form.fields['last_name'].label == 'Last Name' and
            form.fields['username'].label == 'Roll number' and
            form.fields['dob'].label == 'Date of Birth' and
            form.fields['department'].label == 'Department' and
            form.fields['email'].label == 'Email'
        )

    def test_signup_form_required_fields(self):
        form = FacultySignupForm()
        self.assertTrue(
            form.fields['first_name'].required == True and
            form.fields['last_name'].required == True and
            form.fields['dob'].required == True and
            form.fields['department'].required == True and
            form.fields['email'].required == True
        )

    def test_invalid_email_validation(self):

        startTime = timezone.now()
        department = Department.objects.create(name='test department')
        user = User.objects.create(
            username='test',
            email='test@gmail.com'
        )
        faculty = Faculty.objects.create(
            user=user,
            dob=startTime,
            department=department
        )

        form = FacultySignupForm(
            data = {
                'username': 'test1',
                'email': 'test@gmail.com',
                'dob': startTime,
                'department': department
            }
        )
        self.assertFalse(form.is_valid())

    def test_valid_email_validation(self):

        startTime = timezone.now()
        department = Department.objects.create(name='test department')

        form = FacultySignupForm(
            data = {
                'username': 'test',
                'first_name': 'Bob',
                'last_name': 'Davidson',
                'dob': startTime,
                'email': 'test@gmail.com',
                'password1': 'complex1password',
                'password2': 'complex1password',
                'department': department
            }
        )
        self.assertTrue(form.is_valid())
