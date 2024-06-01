#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Charts about the national vaccines data.
@author: riccardomaldini
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from data_extractors.vaccines_regions import benchmark_dict, marche_df
from data_extractors.vaccines_italy import italy_df
from data_extractors.area_names import area_names_dict
from matplotlib.dates import MonthLocator
import utils


def adm_doses_italy(save_image=False, show=False):
    """
    Administration data about Italy.
    """

    # plt.stackplot(data['data_somministrazione'], data['prima_dose'],data['seconda_dose'],
    #               labels=['Prime dosi', 'Seconde dosi'])
    plt.bar(italy_df['data_somministrazione'], italy_df['prima_dose'], label='Prime dosi')
    plt.bar(italy_df['data_somministrazione'], italy_df['seconda_dose'], bottom=italy_df['prima_dose'],
            label='Seconde dosi')

    plt.title("Somministrazioni giornaliere Italia,\ncon distinzione prima dose/richiamo\n")
    plt.gca().xaxis.set_major_locator(MonthLocator())
    plt.gca().xaxis.set_minor_locator(MonthLocator(bymonthday=15))
    plt.gca().xaxis.set_major_formatter(utils.std_date_formatter)
    plt.gca().xaxis.set_minor_formatter(utils.std_date_formatter)
    plt.gcf().autofmt_xdate(which='both')
    plt.grid(True, which='both', axis='both')
    plt.legend(loc='upper left')

    if save_image:
        plt.savefig('./charts/vaccines/dosi_italia.png', dpi=300, transparent=True, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def adm_doses_marche(save_image=False, show=False):
    """
    Administration data about Italy.
    """

    plt.bar(marche_df['data_somministrazione'], marche_df['prima_dose'], label='Prime dosi')
    plt.bar(marche_df['data_somministrazione'], marche_df['seconda_dose'], bottom=marche_df['prima_dose'],
            label='Seconde dosi')

    plt.title("Somministrazioni giornaliere Marche,\ncon distinzione prima dose/richiamo\n")
    plt.gca().xaxis.set_major_locator(MonthLocator())
    plt.gca().xaxis.set_minor_locator(MonthLocator(bymonthday=15))
    plt.gca().xaxis.set_major_formatter(utils.std_date_formatter)
    plt.gca().xaxis.set_minor_formatter(utils.std_date_formatter)
    plt.gcf().autofmt_xdate(which='both')
    plt.grid(True, which='both', axis='both')
    plt.legend(loc='upper left')

    if save_image:
        plt.savefig('./charts/vaccines/dosi_marche.png', dpi=300, transparent=True, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def regional_doses(save_image=False, show=False):
    """
    Comparation between doses administrated in various regions
    """

    for area_code, region_data in benchmark_dict.items():
        rolling_avg_adm = region_data['totale_per_100000_ab'].rolling(7, center=True).mean()
        plt.plot(region_data['data_somministrazione'], rolling_avg_adm, label=area_names_dict[area_code])

    rolling_avg_adm = italy_df['totale_per_100000_ab'].rolling(7, center=True).mean()
    plt.plot(italy_df['data_somministrazione'], rolling_avg_adm, alpha=0.5, linestyle=':',
             label="Italia")

    plt.title('Andamento delle somministrazioni giornaliere\nper 100.000 abitanti, confronto tra le regioni del benchmark\n')
    plt.gca().xaxis.set_major_locator(MonthLocator())
    plt.gca().xaxis.set_minor_locator(MonthLocator(bymonthday=15))
    plt.gca().xaxis.set_major_formatter(utils.std_date_formatter)
    plt.gca().xaxis.set_minor_formatter(utils.std_date_formatter)
    plt.gcf().autofmt_xdate(which='both')
    plt.grid(True, which='both', axis='both')
    plt.legend(loc='upper left')

    if save_image:
        plt.savefig('./charts/vaccines/dosi_per_regioni.png', dpi=300, transparent=True, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()


def immunes_percentage(save_image=False, show=False):
    """
    Computes and plots relations between the population of a place and people that took the second shot.
    """

    for area_code, region_data in benchmark_dict.items():
        plt.plot(region_data['data_somministrazione'], region_data['seconda_dose_totale_storico_su_pop'],
                 label=area_names_dict[area_code])

    plt.plot(italy_df['data_somministrazione'], italy_df['seconda_dose_totale_storico_su_pop'], alpha=0.5, linestyle=':',
             label="Italia")

    plt.title('Percentuale popolazione immunizzata,\nconfronto tra le regioni del benchmark\n')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    plt.gca().xaxis.set_major_locator(MonthLocator())
    plt.gca().xaxis.set_minor_locator(MonthLocator(bymonthday=15))
    plt.gca().xaxis.set_major_formatter(utils.std_date_formatter)
    plt.gca().xaxis.set_minor_formatter(utils.std_date_formatter)
    plt.gcf().autofmt_xdate(which='both')
    plt.grid(True, which='both', axis='both')
    plt.legend(loc='upper left')

    if save_image:
        plt.savefig('./charts/vaccines/immunizzati.png', dpi=300, transparent=True, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()
