# python3 imports
from re import compile as compile_regex
from gettext import gettext as _

# project imports
from wintersdeep_postcode.postcode import Postcode
from wintersdeep_postcode.exceptions.validation_fault import ValidationFault

## A wrapper for validation of standard postcodes
#  @remarks see \ref wintersdeep_postcode.postcode_types.standard_postcode 
class StandardPostcodeValidator(object):



    ## Areas that only have single digit districts (ignoring sub-divisions)
    #  @remarks loaded from JSON file 'standard_postcode_validator.json'
    AreasWithOnlySingleDigitDistricts = []

    ## Checks if a postcode is in an area with only single digit districts and if 
    #  so - that the district specified is only a single digit.
    #  @param cls the type of class that is invoking this method.
    #  @param postcode the postcode to check for conformance to this rule.
    #  @returns True if the postcode violates this rule, else False.
    @classmethod
    def CheckAreasWithOnlySingleDigitDistricts(cls, postcode):
        impacted_by_rule = False
        if postcode.outward_district >= 10:
            single_digit_districts = cls.AreasWithOnlySingleDigitDistricts
            impacted_by_rule = postcode.outward_area in single_digit_districts
        return impacted_by_rule
                


    ## Areas that only have double digit districts (ignoring sub-divisions)
    #  @remarks loaded from JSON file 'standard_postcode_validator.json'
    AreasWithOnlyDoubleDigitDistricts = []
        
    ## Checks if a postcode is in an area with only double digit districts and 
    #  if so - that the district specified has two digits as required.
    #  @param cls the type of class that is invoking this method.
    #  @param postcode the postcode to check for conformance to this rule.
    #  @returns True if the postcode violates this rule, else False.
    @classmethod
    def CheckAreasWithOnlyDoubleDigitDistricts(cls, postcode):
        impacted_by_rule = False
        if postcode.outward_district <= 9:
            double_digit_districts = cls.AreasWithOnlyDoubleDigitDistricts
            impacted_by_rule = postcode.outward_area in double_digit_districts
        return impacted_by_rule
                


    ## Areas that have a district zero.
    #  @remarks loaded from JSON file 'standard_postcode_validator.json'
    AreasWithDistrictZero = []
        
    ## Checks if a postcode has a district zero if it specified one.
    #  @param cls the type of class that is invoking this method.
    #  @param postcode the postcode to check for conformance to this rule.
    #  @returns True if the postcode violates this rule, else False.
    @classmethod
    def CheckAreasWithDistrictZero(cls, postcode):
        impacted_by_rule = False
        if postcode.outward_district == 0:
            areas_with_district_zero = cls.AreasWithDistrictZero
            impacted_by_rule = not postcode.outward_area in areas_with_district_zero
        return impacted_by_rule



    ## Areas that do not have a district 10
    #  @remarks loaded from JSON file 'standard_postcode_validator.json'
    AreasWithoutDistrictTen = []
        
    ## Checks if a postcode has a district ten if it specified one.
    #  @param cls the type of class that is invoking this method.
    #  @param postcode the postcode to check for conformance to this rule.
    #  @returns True if the postcode violates this rule, else False.
    @classmethod
    def CheckAreasWithoutDistrictTen(cls, postcode):
        impacted_by_rule = False
        if postcode.outward_district == 10:
            areas_without_district_ten = cls.AreasWithoutDistrictTen
            impacted_by_rule = postcode.outward_area in areas_without_district_ten
        return impacted_by_rule



    ## Only a few areas have subdivided districts
    #  @remarks loaded from JSON file 'standard_postcode_validator.json'
    AreasWithSubdistricts = {}

    ## If a postcode has subdistricts, check its supposed to.
    #  @param cls the type of class that is invoking this method.
    #  @param postcode the postcode to check for conformance to this rule.
    #  @returns True if the postcode violates this rule, else False.
    @classmethod
    def CheckAreasWithSubdistricts(cls, postcode):
        impacted_by_rule = False
        if postcode.outward_subdistrict:
            areas_with_subdistricts = cls.AreasWithSubdistricts
            impacted_by_rule = not postcode.outward_area in areas_with_subdistricts
            if not impacted_by_rule:
                subdivided_districts_in_area = areas_with_subdistricts[postcode.outward_area]
                if subdivided_districts_in_area:
                    impacted_by_rule = not postcode.outward_district in subdivided_districts_in_area
        return impacted_by_rule

    ## If a postcode has a limited selection of subdistricts, makes sure any set are in scope.
    #  @param cls the type of class that is invoking this method.
    #  @param postcode the postcode to check for conformance to this rule.
    #  @returns True if the postcode violates this rule, else False.
    @classmethod
    def CheckAreasWithSpecificSubdistricts(cls, postcode):
        impacted_by_rule = False
        if postcode.outward_subdistrict:
            areas_with_subdistricts = cls.AreasWithSubdistricts
            subdivided_districts_in_area = areas_with_subdistricts.get(postcode.outward_area, {})
            specific_subdistrict_codes = subdivided_districts_in_area.get(postcode.outward_district, None)
            impacted_by_rule = specific_subdistrict_codes and \
                not postcode.outward_subdistrict in specific_subdistrict_codes
        return impacted_by_rule



    ## Charactesr that are not used in the first position.
    #  @remarks loaded from JSON file 'standard_postcode_validator.json'
    FirstPositionExcludes = []
    
    ## Checks that a postcode does not include usued characters in the first postition.
    #  @param cls the type of class that is invoking this method.
    #  @param postcode the postcode to check for conformance to this rule.
    #  @returns True if the postcode violates this rule, else False.
    @classmethod
    def CheckFirstPositionExcludes(cls, postcode):
        first_postion_char = postcode.outward_area[0]
        impacted_by_rule = first_postion_char in cls.FirstPositionExcludes
        return impacted_by_rule



    ## Charactesr that are not used in the second position.
    #  @remarks loaded from JSON file 'standard_postcode_validator.json'
    SecondPositionExcludes = []
    
    ## Checks that a postcode does not include unused characters in the second postition.
    #  @param cls the type of class that is invoking this method.
    #  @param postcode the postcode to check for conformance to this rule.
    #  @returns True if the postcode violates this rule, else False.
    @classmethod
    def CheckSecondPositionExcludes(cls, postcode):
        impacted_by_rule = False
        if len(postcode.outward_area) > 1:
            second_postion_char = postcode.outward_area[1]
            impacted_by_rule = second_postion_char in cls.SecondPositionExcludes
        return impacted_by_rule



    ## Charactesr that are used in the third apha position (for single digit areas).
    #  @remarks loaded from JSON file 'standard_postcode_validator.json'
    SingleDigitAreaSubdistricts = []
    
    ## Checks that a postcode does not include unused subdistricts for single digit areas.
    #  @param cls the type of class that is invoking this method.
    #  @param postcode the postcode to check for conformance to this rule.
    #  @returns True if the postcode violates this rule, else False.
    @classmethod
    def CheckSingleDigitAreaSubdistricts(cls, postcode):
        impacted_by_rule = False
        if postcode.outward_subdistrict:
            if len(postcode.outward_area) == 1:
                allowed_subdistricts = cls.SingleDigitAreaSubdistricts
                subdistrict = postcode.outward_subdistrict
                impacted_by_rule = not subdistrict in allowed_subdistricts
        return impacted_by_rule

    ## Charactesr that are used in the fourth apha position (for double digit areas).
    #  @remarks loaded from JSON file 'standard_postcode_validator.json'
    DoubleDigitAreaSubdistricts = []
    
    ## Checks that a postcode does not include unused subdistricts for double digit areas.
    #  @param cls the type of class that is invoking this method.
    #  @param postcode the postcode to check for conformance to this rule.
    #  @returns True if the postcode violates this rule, else False.
    @classmethod
    def CheckDoubleDigitAreaSubdistricts(cls, postcode):
        impacted_by_rule = False
        if postcode.outward_subdistrict:
            if len(postcode.outward_area) == 2:
                allowed_subdistricts = cls.DoubleDigitAreaSubdistricts
                subdistrict = postcode.outward_subdistrict
                impacted_by_rule = not subdistrict in allowed_subdistricts
        return impacted_by_rule



    ## Charactesr that are not used in the unit string.
    #  @remarks loaded from JSON file 'standard_postcode_validator.json'
    UnitExcludes = []
    
    ## Checks that a postcode does not include characters in the first character of the unit string that are unused.
    #  @remarks we check the first/second unit character seperately to provide more comprehensive errors.
    #  @param cls the type of class that is invoking this method.
    #  @param postcode the postcode to check for conformance to this rule.
    #  @returns True if the postcode violates this rule, else False.
    @classmethod
    def CheckFirstUnitCharacterExcludes(cls, postcode):
        character = postcode.inward_unit[0]
        impacted_by_rule = character in cls.UnitExcludes
        return impacted_by_rule
    
    ## Checks that a postcode does not include characters in the second character of the unit string that are unused.
    #  @remarks we check the first/second unit character seperately to provide more comprehensive errors.
    #  @param cls the type of class that is invoking this method.
    #  @param postcode the postcode to check for conformance to this rule.
    #  @returns True if the postcode violates this rule, else False.
    @classmethod
    def CheckSecondUnitCharacterExcludes(cls, postcode):
        character = postcode.inward_unit[1]
        impacted_by_rule = character in cls.UnitExcludes
        return impacted_by_rule


## Loads various static members used for validation of standard postcodes from
#  a JSON file - this is expected to be co-located with this class.
def load_validator_params_from_json():
    
    from json import load
    from os.path import dirname, join
    
    json_configuration_file = join( dirname(__file__), "standard_postcode_validator.json" )
    
    with open(json_configuration_file, 'r') as file_handle:
        config_json = load(file_handle)

    StandardPostcodeValidator.AreasWithDistrictZero = config_json['has-district-zero']
    StandardPostcodeValidator.AreasWithoutDistrictTen = config_json['no-district-ten']
    StandardPostcodeValidator.AreasWithOnlyDoubleDigitDistricts = config_json['double-digit-districts']
    StandardPostcodeValidator.AreasWithOnlySingleDigitDistricts = config_json['single-digit-districts']
    StandardPostcodeValidator.SingleDigitAreaSubdistricts = config_json['single-digit-area-subdistricts']
    StandardPostcodeValidator.DoubleDigitAreaSubdistricts = config_json['double-digit-area-subdistricts']
    StandardPostcodeValidator.SecondPositionExcludes = config_json['second-position-excludes']
    StandardPostcodeValidator.FirstPositionExcludes = config_json['first-position-excludes']
    StandardPostcodeValidator.UnitExcludes = config_json['unit-excludes']
    
    subdivision_map = config_json["subdivided-districts"]
    StandardPostcodeValidator.AreasWithSubdistricts = {  k: { 
        int(k1): v1 for k1, v1 in v.items()
    } for k, v in subdivision_map.items() }


load_validator_params_from_json()

if __name__ == "__main__":
    
    ##
    ##  If this is the main entry point - someone might be a little lost?
    ##

    print(f"{__file__} ran, but doesn't do anything on its own.")
    print(f"Check 'https://www.github.com/wintersdeep/wintersdeep_postcode' for usage.")