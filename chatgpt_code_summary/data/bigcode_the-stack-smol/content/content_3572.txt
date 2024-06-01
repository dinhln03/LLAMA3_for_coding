from django.db import models

class alumno(models.Model):
    
    alum_id = models.AutoField(primary_key=True)
    alum_nom = models.CharField(max_length=100, help_text="Nombre del alumno", unique=True)
    alum_ape = models.CharField(max_length=100, help_text="Apellido del alumno", unique=True)
    
    def __str__(self):
        return '{}'.format(self.alum_nom, self.alum_ape)
    
    class Meta:
        db_table = "alumno"
        verbose_name_plural="alumnos"
    
class curso(models.Model):
    
    cur_id = models.AutoField(primary_key=True)
    cur_nom = models.CharField(max_length=100, help_text="Nombre del alumno", unique=True)
    
    def __str__(self):
        return '{}'.format(self.cur_nom)
    
    class Meta:
        db_table = "curso"
        verbose_name_plural="cursos"
  
    
class alm_cur(models.Model):
    
    almcur_id = models.AutoField(primary_key=True)
    alum_id = models.ForeignKey(alumno, on_delete=models.CASCADE)
    cur_id = models.ForeignKey(curso, on_delete=models.CASCADE)
    
    def __str__(self):
        return '{}'.format(self.almcur_id)
    
    class Meta:
        db_table = "alm_cur"
        verbose_name_plural="alm_cursos"
        
    

class asistencia(models.Model):
    
    asis_id = models.AutoField(primary_key=True)
    asis_fecha = models.Date()
    asis_est = models.BooleanField(default=False)
    almcur_id = models.ForeignKey(alm_cur, on_delete=models.CASCADE)
    
    def __str__(self):
        return '{}'.format(self.asis_fecha, self.asis_est)
    
    
    class Meta:
        db_table = "asistencia"
        verbose_name_plural="asistencias"


