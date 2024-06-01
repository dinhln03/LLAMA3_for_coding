from django.db import models
from osoba.models import ServiseID, Company
from django.utils.translation import gettext as _


class SHPK(models.Model):
    name = models.CharField(max_length=512, verbose_name=_('Name'))
    short_name = models.CharField(max_length=512, verbose_name=_('Short name'))

    def __str__(self):
        return self.name[:50]

    class Meta:
        verbose_name = _('SHPK')
        verbose_name_plural = _('SHPK')



class ZvannyaName(models.Model):
    zv_id = models.AutoField(primary_key=True)
    zv_name = models.TextField()
    zv_short_name = models.CharField(max_length=20)

    def __str__(self):
        return '{}__{}'.format(self.zv_id, self.zv_name)

    class Meta:
        managed = False
        db_table = 'zvannya_name'



class Staff(models.Model):
    # Штатка
    #порядковий номер в штатці
    unicum = models.PositiveBigIntegerField(verbose_name= _('Unic number'), blank=True)
    company = models.ForeignKey(Company, on_delete=models.CASCADE, blank=True, verbose_name= _('Company'))
    name = models.CharField(max_length=512, verbose_name=_('Name'))
    shpk = models.ForeignKey(SHPK, on_delete=models.CASCADE, blank=True, verbose_name= _('shpk'))
    ocoba = models.ForeignKey(ServiseID, on_delete=models.CASCADE, blank=True, verbose_name= _('ocoba'), null=True)
    vos = models.CharField(max_length=512, verbose_name= _('VOS'))
    poz = models.CharField(max_length=512, verbose_name= _('pozyvnyy'), blank=True)
    salary = models.PositiveBigIntegerField(verbose_name= _('salary'), blank=True)
    tariff_category = models.PositiveBigIntegerField(verbose_name= _('tariff category'), blank=True)
    vacant = models.BooleanField(verbose_name= _('Vacant'), blank=True, null=True, default=True)

    def __str__(self):
        return self.name[:50]

    class Meta:
        verbose_name = _('Staff')
        verbose_name_plural = _('Staff')



class Adresa(models.Model):
    adr_id = models.AutoField(primary_key=True)
    adr_n_id = models.IntegerField()
    adresa = models.CharField(max_length=360)

    class Meta:
        managed = False
        db_table = 'adresa'





class Nakaz(models.Model):
    nak_id = models.AutoField(primary_key=True)
    nak_n_id = models.IntegerField()
    nak_status_id = models.IntegerField()
    zvidky = models.IntegerField()
    kudy = models.IntegerField()
    nak_data = models.DateField( blank=True, null=True)
    nak_nomer = models.IntegerField()
    povern = models.DateField( blank=True, null=True)
    tmp = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'nakaz'


class NakazNomer(models.Model):
    n_nak_id = models.AutoField(primary_key=True)
    n_nak_data = models.DateField( blank=True, null=True)
    n_nak_nomer = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'nakaz_nomer'


class NakazPlace(models.Model):
    nak_place_id = models.AutoField(primary_key=True)
    nak_place_name = models.CharField(max_length=120)

    class Meta:
        managed = False
        db_table = 'nakaz_place'


class PosadaName(models.Model):
    pos_id = models.AutoField(primary_key=True)
    pos_name = models.TextField()

    def __str__(self):
        return '{}__{}'.format(self.pos_id, self.pos_name[:50])

    class Meta:
        managed = False
        db_table = 'posada_name'


class PidrozdilName(models.Model):
    p_id = models.AutoField(primary_key=True)
    por_nomer = models.IntegerField()
    p_name = models.TextField()
    p_short_name = models.CharField(max_length=32)
    p_full_name = models.CharField(max_length=200)
    active = models.IntegerField()

    def __str__(self):
        return '{}__{}'.format(self.p_id, self.p_name[:50])

    class Meta:
        managed = False
        db_table = 'pidrozdil_name'



class Shtatka(models.Model):
    pos_id = models.AutoField(primary_key=True)
    p = models.ForeignKey(PidrozdilName, to_field='p_id', on_delete=models.PROTECT, related_name='+' )
    sh = models.ForeignKey(PosadaName, to_field='pos_id', on_delete=models.PROTECT, related_name='+' )
    zv_sh = models.ForeignKey(ZvannyaName, to_field='zv_id', on_delete=models.PROTECT, related_name='+' )
    dopusk = models.CharField(max_length=1)
    vos = models.CharField(max_length=12)
    oklad = models.CharField(max_length=12)
    vidsotok = models.IntegerField()
    nomer_kniga = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'shtatka'

    def __str__(self):
        return '{}__{}'.format(self.pos_id, self.sh)



class OsvitaName(models.Model):
    osv_name_id = models.AutoField(primary_key=True)
    osv_name = models.CharField(max_length=100)

    def __str__(self):
        return '{}__{}'.format(self.osv_name_id, self.osv_name)

    class Meta:
        managed = False
        db_table = 'osvita_name'


class SimStanName(models.Model):
    s_stan_name_id = models.AutoField(primary_key=True)
    s_stan_name = models.CharField(max_length=30)

    def __str__(self):
        return '{}__{}'.format(self.s_stan_name_id, self.s_stan_name)


    class Meta:
        managed = False
        db_table = 'sim_stan_name'


class StatsName(models.Model):
    s_stats_name_id = models.AutoField(primary_key=True)
    s_stats_name = models.CharField(max_length=1)

    def __str__(self):
        return '{}__{}'.format(self.s_stats_name_id, self.s_stats_name)

    class Meta:
        managed = False
        db_table = 'stats_name'


class StatusName(models.Model):
    s_id = models.AutoField(primary_key=True)
    s_name = models.CharField(max_length=128)

    def __str__(self):
        return '{}__{}'.format(self.s_id, self.s_name)

    class Meta:
        managed = False
        db_table = 'status_name'


class Name(models.Model):
    n_id= models.AutoField(primary_key=True)
    name = models.TextField()
    short_name = models.TextField()
    pseudo = models.CharField(max_length=128)
    zv = models.ForeignKey(ZvannyaName, to_field='zv_id', on_delete=models.PROTECT, related_name='+' )
    data_zv = models.CharField(max_length=100)
    pos = models.ForeignKey(Shtatka, to_field='pos_id', on_delete=models.PROTECT, related_name='+' )
    pos_id_old = models.IntegerField(null=True, blank=True)
    p_id = models.IntegerField() #wtf?
    kontr = models.IntegerField(null=True, blank=True)
    data_narod = models.DateField( blank=True, null=True)
    adresa_nar = models.CharField(max_length=200)
    data_mob = models.DateField( blank=True, null=True)
    vijskomat = models.CharField(max_length=100)
    data_zarah = models.DateField( blank=True, null=True)
    nomer_nakazu_ok = models.CharField(max_length=10)
    data_nakazu_ok = models.DateField( blank=True, null=True)
    chiy_nakaz = models.CharField(max_length=50)
    kontrakt = models.DateField( blank=True, null=True)
    kontrakt_strok = models.CharField(max_length=50)
    kontrakt_zak = models.DateField( blank=True, null=True)
    nomer_nakazu = models.IntegerField()#wtf?
    data_zviln = models.DateField( blank=True, null=True)
    nomer_nakazu_zviln = models.IntegerField()#wtf?
    nomer_pasp = models.CharField(max_length=100)
    code_nomer = models.CharField(max_length=10)
    voen_nomer = models.CharField(max_length=25)
    grupa_krovi = models.CharField(max_length=15)
    osvita = models.ForeignKey(OsvitaName, to_field='osv_name_id', on_delete=models.PROTECT, related_name='+' )
    specialnist = models.CharField(max_length=500)
    zvp = models.CharField(max_length=100)
    fahova = models.CharField(max_length=100)
    liderstvo = models.CharField(max_length=100)
    perem = models.CharField(max_length=50)
    persh_kontr = models.CharField(max_length=50)
    ozdor = models.CharField(max_length=50)
    mdspp = models.CharField(max_length=50)
    sim_stan = models.ForeignKey(SimStanName, to_field='s_stan_name_id', on_delete=models.PROTECT, related_name='+' )
    stats = models.ForeignKey(StatsName, to_field='s_stats_name_id', on_delete=models.PROTECT, related_name='+' )
    status = models.ForeignKey(StatusName, to_field='s_id', on_delete=models.PROTECT, related_name='+' )
    status2 = models.IntegerField()
    notes = models.TextField()
    notes1 = models.TextField()

    def __str__(self):
        return self.name[:50]

    class Meta:
        managed = False
        db_table = 'name'


class Peremish(models.Model):
    perem_id = models.AutoField(primary_key=True)
    perem_n_id = models.IntegerField()
    perem_status_id = models.IntegerField()
    zvidky = models.IntegerField()
    kudy = models.IntegerField()
    perem_data = models.DateField( blank=True, null=True)
    nakaz_id = models.IntegerField()
    povern = models.DateField( blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'peremish'


class Phones(models.Model):
    ph_id = models.AutoField(primary_key=True)
    n_id = models.IntegerField()
    ph_nomer = models.TextField()

    class Meta:
        managed = False
        db_table = 'phones'


class PidrozdilId(models.Model):
    p_id = models.AutoField(primary_key=True)
    p_parent_id = models.IntegerField()
    isparent = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'pidrozdil_id'



class Priznach(models.Model):
    prizn_id = models.AutoField(primary_key=True)
    prizn_n_id = models.IntegerField()
    prizn_data = models.DateField( blank=True, null=True)
    prizn_pos_id = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'priznach'


class PriznachOld(models.Model):
    prizn_id = models.AutoField(primary_key=True)
    prizn_n_id = models.IntegerField()
    prizn_data = models.DateField( blank=True, null=True)
    prizn_pos_id = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'priznach_old'


class PriznachOld2(models.Model):
    prizn_id = models.AutoField(primary_key=True)
    prizn_n_id = models.IntegerField()
    prizn_data = models.DateField( blank=True, null=True)
    prizn_pos_id = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'priznach_old_2'


class Ridny(models.Model):
    rid_id = models.AutoField(primary_key=True)
    rid_n_id = models.IntegerField()
    rid_name_id = models.IntegerField()
    rid_name = models.CharField(max_length=200)
    rid_data_nar = models.DateField( blank=True, null=True)
    rid_phone = models.IntegerField()
    rid_notes = models.CharField(max_length=500)

    class Meta:
        managed = False
        db_table = 'ridny'


class RidnyName(models.Model):
    rid_name_id = models.AutoField(primary_key=True)
    rid_name_name = models.CharField(max_length=50)

    class Meta:
        managed = False
        db_table = 'ridny_name'


class ShtatkaOld(models.Model):
    pos_id = models.AutoField(primary_key=True)
    p_id = models.IntegerField()
    sh_id = models.IntegerField()
    zv_sh_id = models.IntegerField()
    vos = models.CharField(max_length=12)
    oklad = models.CharField(max_length=12)
    nomer_kniga = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'shtatka_old'


class ShtatkaOld2(models.Model):
    pos_id = models.AutoField(primary_key=True)
    p_id = models.IntegerField()
    sh_id = models.IntegerField()
    zv_sh_id = models.IntegerField()
    vos = models.CharField(max_length=12)
    oklad = models.CharField(max_length=12)
    nomer_kniga = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'shtatka_old_2'


class Table32(models.Model):
    col_1 = models.CharField(db_column='COL 1', max_length=10, blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    col_2 = models.IntegerField(db_column='COL 2', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.

    class Meta:
        managed = False
        db_table = 'table 32'


class Table35(models.Model):
    col_1 = models.CharField(db_column='COL 1', max_length=10, blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    col_2 = models.IntegerField(db_column='COL 2', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.

    class Meta:
        managed = False
        db_table = 'table 35'


class Temp(models.Model):
    number_1 = models.IntegerField(db_column='1')  # Field renamed because it wasn't a valid Python identifier.
    number_2 = models.TextField(db_column='2')  # Field renamed because it wasn't a valid Python identifier.

    class Meta:
        managed = False
        db_table = 'temp'


class Tmp(models.Model):
    number_1 = models.IntegerField(db_column='1')  # Field renamed because it wasn't a valid Python identifier.
    number_2 = models.TextField(db_column='2')  # Field renamed because it wasn't a valid Python identifier.

    class Meta:
        managed = False
        db_table = 'tmp'


class Vysluga(models.Model):
    vys_id = models.AutoField(primary_key=True)
    vys_n_id = models.IntegerField()
    vys_data_mob = models.DateField( blank=True, null=True)
    vys_data_zvil = models.DateField( blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'vysluga'


class VyslugaNormy(models.Model):
    rokiv = models.IntegerField()
    nadbavka = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'vysluga_normy'


class VyslugaZv(models.Model):
    vys_zv_id = models.AutoField(primary_key=True)
    vys_zv_n_id = models.IntegerField()
    data_zv = models.DateField( blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'vysluga_zv'


class ZbrStatusName(models.Model):
    zbr_status_id = models.AutoField(primary_key=True)
    zbr_status_name = models.CharField(max_length=20)

    class Meta:
        managed = False
        db_table = 'zbr_status_name'


class Zbroya(models.Model):
    zbr_id = models.AutoField(primary_key=True)
    zbr_type = models.IntegerField()
    nomer = models.CharField(max_length=128)
    n_id = models.IntegerField()
    magazin = models.IntegerField()
    zbr_status = models.IntegerField()
    zbr_note = models.CharField(max_length=256)

    class Meta:
        managed = False
        db_table = 'zbroya'


class ZbroyaAll(models.Model):
    zbr_type = models.IntegerField()
    nomer = models.CharField(max_length=128)
    rota = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'zbroya_all'


class ZbroyaName(models.Model):
    zbr_id = models.AutoField(primary_key=True)
    zbr_name = models.CharField(max_length=128)

    class Meta:
        managed = False
        db_table = 'zbroya_name'


class ZbroyaSklad(models.Model):
    zbr_type = models.IntegerField()
    nomer = models.CharField(max_length=256)

    class Meta:
        managed = False
        db_table = 'zbroya_sklad'


class ZvGrupaName(models.Model):
    zv_gr_id = models.AutoField(primary_key=True)
    zv_gr_name = models.CharField(max_length=20)

    class Meta:
        managed = False
        db_table = 'zv_grupa_name'


class ZvannyaId(models.Model):
    zv_id = models.IntegerField(unique=True)
    zv_gr_id = models.IntegerField()
    zv_okl = models.IntegerField()

    class Meta:
        managed = False
        db_table = 'zvannya_id'


class ZvilnComent(models.Model):
    zv_com_id = models.AutoField(primary_key=True)
    zv_com_n_id = models.IntegerField()
    zv_coment = models.CharField(max_length=500)

    class Meta:
        managed = False
        db_table = 'zviln_coment'


class Kontrakt(models.Model):
    kontrakt_com_id = models.AutoField(primary_key=True)
    kontrakt_com_n = models.ForeignKey(Name, to_field='n_id', on_delete=models.PROTECT, related_name='+')# models.IntegerField()
    kontrakt_date = models.DateField( blank=True, null=True)
    kontrakt_srok = models.IntegerField()
    kontrakt_zak = models.DateField( blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'kontrakt'
