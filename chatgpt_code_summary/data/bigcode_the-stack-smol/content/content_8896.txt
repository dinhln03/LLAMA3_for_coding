from xml.dom import minidom

from django.utils.datastructures import MultiValueDict
from django import forms
from django.utils.html import format_html, mark_safe
from django.forms.utils import flatatt

class SelectMultipleSVG(forms.SelectMultiple):
    class Media:
        js = ('django_svgselect.js',)

    def __init__(self, svg):
        super(SelectMultipleSVG, self).__init__()
        # TODO: Add some validation here?
        self.svg = svg

    def render(self, name, value, attrs=None, choices=()):
        svg = minidom.parse(self.svg)

        if value is None:
            value = []
        final_attrs = self.build_attrs(attrs, name=name)

        output = [format_html('<select multiple="multiple"{}>', flatatt(final_attrs))]
        options = self.render_options(choices, value)
        if options:
            output.append(options)
        output.append('</select>')
        output.append("<div id='%s-svg'>" % final_attrs['id'])
        output.append(svg.toxml())
        output.append("</div>")

        output.append("<script language='javascript'>document.getElementById('%s').convertToSvg('%s-svg');</script>" %
                      (final_attrs['id'], final_attrs['id']))

        return mark_safe('\n'.join(output))

    def value_from_datadict(self, data, files, name):
        if isinstance(data, MultiValueDict):
            return data.getlist(name)
        return data.get(name)