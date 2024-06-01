#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# (C)Eduardo Ribeiro - 1600820


class Contract:
    id = 0
    school_code = 0
    school_name = ""
    n_contract = 0
    n_hours_per_week = 0
    contract_end_date = ""
    application_deadline = ""
    recruitment_group = ""
    county = ""
    district = ""
    class_project = ""
    qualifications = ""

    def __init__(
        self,
        id,
        school_code,
        school_name,
        n_contract,
        n_hours_per_week,
        contract_end_date,
        application_deadline,
        recruitment_group,
        county,
        district,
        class_project,
        qualifications,
    ):
        self.id = id
        self.school_code = school_code
        self.school_name = school_name
        self.n_contract = n_contract
        self.n_hours_per_week = n_hours_per_week
        self.contract_end_date = contract_end_date
        self.application_deadline = application_deadline
        self.recruitment_group = recruitment_group
        self.county = county
        self.district = district
        self.class_project = class_project
        self.qualifications = qualifications
