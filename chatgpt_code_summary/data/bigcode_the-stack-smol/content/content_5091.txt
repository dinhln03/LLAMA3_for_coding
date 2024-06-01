from __future__ import annotations

from abc import abstractmethod, ABC
from decimal import Decimal
from enum import Enum
from typing import Dict, cast

import numpy as np

# A few extra general types
from slim.simulation.lice_population import LicePopulation, GenoDistrib, GenoTreatmentValue,\
    Alleles, GenoTreatmentDistrib

Money = Decimal


class Treatment(Enum):
    """
    A stub for treatment types
    TODO: add other treatments here
    """
    EMB = 0
    THERMOLICER = 1


class GeneticMechanism(Enum):
    """
    Genetic mechanism to be used when generating egg genotypes
    """
    DISCRETE = 1
    MATERNAL = 2


class HeterozygousResistance(Enum):
    """
    Resistance in a monogenic, heterozygous setting.
    """
    DOMINANT = 1
    INCOMPLETELY_DOMINANT = 2
    RECESSIVE = 3


TreatmentResistance = Dict[HeterozygousResistance, float]


class TreatmentParams(ABC):
    """
    Abstract class for all the treatments
    """
    name = ""

    def __init__(self, payload):
        self.quadratic_fish_mortality_coeffs = np.array(payload["quadratic_fish_mortality_coeffs"])
        self.effect_delay: int = payload["effect_delay"]
        self.application_period: int = payload["application_period"]

    @staticmethod
    def parse_pheno_resistance(pheno_resistance_dict: dict) -> TreatmentResistance:
        return {HeterozygousResistance[key.upper()]: val for key, val in pheno_resistance_dict.items()}

    def __get_mortality_pp_increase(self, temperature: float, fish_mass: float) -> float:
        """Get the mortality percentage point difference increase.

        :param temperature: the temperature in Celsius
        :param fish_mass: the fish mass (in grams)
        :returns: Mortality percentage point difference increase
        """
        # TODO: is this the right way to solve this?
        fish_mass_indicator = 1 if fish_mass > 2000 else 0

        input = np.array([1, temperature, fish_mass_indicator, temperature ** 2, temperature * fish_mass_indicator,
                          fish_mass_indicator ** 2])
        return max(float(self.quadratic_fish_mortality_coeffs.dot(input)), 0)

    @abstractmethod
    def delay(self, average_temperature: float):  # pragma: no cover
        """
        Delay before treatment should have a noticeable effect
        """

    @staticmethod
    def get_allele_heterozygous_trait(alleles: Alleles):
        """
        Get the allele heterozygous type
        """
        # should we move this?
        if 'A' in alleles:
            if 'a' in alleles:
                trait = HeterozygousResistance.INCOMPLETELY_DOMINANT
            else:
                trait = HeterozygousResistance.DOMINANT
        else:
            trait = HeterozygousResistance.RECESSIVE
        return trait

    @abstractmethod
    def get_lice_treatment_mortality_rate(
            self, lice_population: LicePopulation, temperature: float) -> GenoTreatmentDistrib:
        """
        Calculate the mortality rates of this treatment
        """

    def get_fish_mortality_occurrences(
            self,
            temperature: float,
            fish_mass: float,
            num_fish: float,
            efficacy_window: float,
            mortality_events: int
    ):
        """Get the number of fish that die due to treatment

        :param temperature: the temperature of the cage
        :param num_fish: the number of fish
        :param fish_mass: the average fish mass (in grams)
        :param efficacy_window: the length of the efficacy window
        :param mortality_events: the number of fish mortality events to subtract from
        """
        predicted_pp_increase = self.__get_mortality_pp_increase(temperature, fish_mass)

        mortality_events_pp = 100 * mortality_events / num_fish
        predicted_deaths = ((predicted_pp_increase + mortality_events_pp) * num_fish / 100) \
                           - mortality_events
        predicted_deaths /= efficacy_window

        return predicted_deaths


class ChemicalTreatment(TreatmentParams):
    """Trait for all chemical treatments"""
    def __init__(self, payload):
        super().__init__(payload)
        self.pheno_resistance = self.parse_pheno_resistance(payload["pheno_resistance"])
        self.price_per_kg = Money(payload["price_per_kg"])

        self.durability_temp_ratio: float = payload["durability_temp_ratio"]


class ThermalTreatment(TreatmentParams):
    """Trait for all thermal-based treatments"""
    def __init__(self, payload):
        super().__init__(payload)
        self.price_per_application = Money(payload["price_per_application"])
        # NOTE: these are currently unused
        # self.exposure_temperature: float = payload["exposure_temperature"]
        # self.exposure_length: float = payload["efficacy"]


class EMB(ChemicalTreatment):
    """Emamectin Benzoate"""
    name = "EMB"

    def delay(self, average_temperature: float):
        return self.durability_temp_ratio / average_temperature

    def get_lice_treatment_mortality_rate(self, lice_population: LicePopulation, _temperature=None):
        susceptible_populations = [lice_population.geno_by_lifestage[stage] for stage in
                                   LicePopulation.susceptible_stages]
        num_susc_per_geno = GenoDistrib.batch_sum(susceptible_populations)

        geno_treatment_distrib = {geno: GenoTreatmentValue(0.0, 0) for geno in num_susc_per_geno}

        for geno, num_susc in num_susc_per_geno.items():
            trait = self.get_allele_heterozygous_trait(geno)
            susceptibility_factor = 1.0 - self.pheno_resistance[trait]
            geno_treatment_distrib[geno] = GenoTreatmentValue(susceptibility_factor, cast(int, num_susc))

        return geno_treatment_distrib


class Thermolicer(ThermalTreatment):
    name = "Thermolicer"

    def delay(self, _):
        return 1 # effects noticeable the next day

    def get_lice_treatment_mortality_rate(
            self, lice_population: LicePopulation, temperature: float) -> GenoTreatmentDistrib:
        if temperature >= 12:
            efficacy = 0.8
        else:
            efficacy = 0.99

        susceptible_populations = [lice_population.geno_by_lifestage[stage] for stage in
                                   LicePopulation.susceptible_stages]
        num_susc_per_geno = cast(GenoDistrib, GenoDistrib.batch_sum(susceptible_populations))

        geno_treatment_distrib = {geno: GenoTreatmentValue(efficacy, cast(int, num_susc))
                                  for geno, num_susc in num_susc_per_geno.items()}
        return geno_treatment_distrib
