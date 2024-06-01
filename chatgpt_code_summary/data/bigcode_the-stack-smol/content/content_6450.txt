from django.db import models


class Produto(models.Model):
    descricao = models.CharField(max_length=30, null=False, blank='False')
    preco = models.DecimalField(max_digits=5, decimal_places=2)
    estoque = models.IntegerField()

    def __str__(self):
        return self.descricao + ' ' + str(self.preco)
