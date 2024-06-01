"""
This module used for serializing data

CategorySchema - data from Category model
VacancySchema - data from Vacancy model
"""
# pylint: disable=too-many-ancestors
# pylint: disable=missing-class-docstring
# pylint: disable=too-few-public-methods

from app import ma
from app.models.model import Category, Vacancy


class CategorySchema(ma.SQLAlchemyAutoSchema):
    """
    Used for serialize Category data
    """
    class Meta:
        model = Category
        fields = ("name", )


class VacancySchema(ma.SQLAlchemyAutoSchema):
    """
    Used for serialize Vacancy data
    """
    class Meta:
        model = Vacancy
        fields = ("name", "salary", "info", "contacts")
        ordered = True


categories_schema = CategorySchema(many=True)
vacancies_schema = VacancySchema(many=True)
