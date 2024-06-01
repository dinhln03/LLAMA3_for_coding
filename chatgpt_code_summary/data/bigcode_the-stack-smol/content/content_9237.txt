from django.db import models
from mptt.models import MPTTModel, TreeForeignKey


class Factor(models.Model):
    group = models.TextField()
    factor = models.TextField()
    note = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ('-group', 'factor',)
        
    def __str__(self):
        txt = 'Group ' + str(self.group) + ' ' + str(self.factor)
        return(txt)


class Element(MPTTModel):
    factor = models.ForeignKey(Factor, to_field='id', on_delete=models.CASCADE, related_name='element_factor')
    value = models.TextField()
    parent = TreeForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='element_child')
    note = models.TextField(blank=True, null=True)

    #class MPTTMeta:
    #    order_insertion_by = ['factor']
   
    class Meta:
        ordering = ('-id', '-factor')

    def __str__(self):
        return(str(self.factor) + ' ' + str(self.value))

    #def get_absolute_url(self):
    #    return(reverse('element:element_detail', args=[self.id]))




