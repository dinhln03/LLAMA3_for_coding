from django.db import models
from django.utils.translation import ugettext_lazy as _
from filer.models.imagemodels import Image

from fractions import Fraction
import exifread

class ExifData(models.Model):
    class Meta:
        verbose_name = _('EXIF Data')
        verbose_name_plural = _('EXIF data')

    image = models.OneToOneField(
        Image,
        verbose_name=_('Image'),
    )

    focal_length = models.CharField(
        max_length=100,
        verbose_name=_('Focal length'),
        blank=True,
    )

    iso = models.CharField(
        max_length=100,
        verbose_name=_('ISO'),
        blank=True,
    )

    fraction = models.CharField(
        max_length=100,
        verbose_name=_('Fraction'),
        blank=True,
    )

    exposure_time = models.CharField(
        max_length=100,
        verbose_name=_('Exposure time'),
        blank=True,
    )

    def save(self, *args, **kwargs):
        if self.pk is None:
            self._read_exif(self.image.file)

        super(ExifData, self).save(*args, **kwargs)

    @property
    def as_list(self):
        tmp = (self.focal_length, self.fraction, self.exposure_time, self.iso)
        return filter(lambda x: x, tmp)

    def _read_exif(self, file):
        result = {}

        try:
            file.open('rb')

            # read tags
            tags = exifread.process_file(file)

            # get necessary tags
            self.focal_length = self._get_and_format(tags,
                                                     "EXIF FocalLength",
                                                     "%gmm", lambda s: Fraction(s))
            self.iso = self._get_and_format(tags,
                                            "EXIF ISOSpeedRatings", "ISO %d",
                                            lambda s: int(s))
            self.fraction = self._get_and_format(tags, "EXIF FNumber", "f/%g",
                                                 lambda s: float(Fraction(s)))

            # format exposure time (fraction or float)
            exposure_time = self._get_and_format(tags, "EXIF ExposureTime",
                                                 None, lambda s: Fraction(s))
            exposure_time_str = ""
            if exposure_time:
                if exposure_time >= 1:
                    exposure_time_str = "%gs" % exposure_time
                else:
                    exposure_time_str = "%ss" % str(exposure_time)

            self.exposure_time = exposure_time_str
        except IOError as e:
            pass

    def _get_and_format(self, tags, key, format, convertfunc):
        """
        Gets element with "key" from dict "tags". Converts this data with
        convertfunc and inserts it into the formatstring "format".

        If "format" is None, the data is returned without formatting, conversion
        is done.

        It the key is not in the dict, the empty string is returned.
        """
        data = tags.get(key, None)
        if data:
            data = convertfunc(str(data))
            if format:
                return format % data
            return data
        return ""
