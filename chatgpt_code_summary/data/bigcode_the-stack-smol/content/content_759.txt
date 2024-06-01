import mysql.connector
import json
import os
import requests


def getAllFindings(host, database, user, password, table, where):
    db = mysql.connector.connect(host=host, database=database, user=user, password=password)
    cursor = db.cursor()
    cursor.execute("SELECT distinct findingCode, specimenOrganCode FROM " + table + " " + where)
    return cursor.fetchall()


def getDrugs(api, filename):
    if filename is None:
        drugs = getDrugsMapping(api)
    else:
        if os.path.isfile(filename):
            with open(filename, 'r') as drug_file:
                drugs = json.loads(drug_file.read())
        else:
            drugs = getDrugsMapping(api)
            with open(filename, 'w') as drug_file:
                drug_file.write(json.dumps(drugs))
    return drugs


def getDrugsMapping(api):
    result = {}
    clinicalCompounds = getClinicalCompounds(api)
    preclinicalCompounds = getPreclinicalCompounds(api)

    # iterate over the clinical and preclinical compounds and match them om inchiKey
    for clinicalCompound in clinicalCompounds:
        for preclinicalCompound in preclinicalCompounds:
            if (clinicalCompound['inchiKey'] is not None) and (clinicalCompound['inchiKey'] == preclinicalCompound['inchiKey']):
                inchiKey = clinicalCompound['inchiKey']
                if inchiKey not in result:
                    result[inchiKey] = {
                        'inchiKey': inchiKey,
                        'clinicalName': clinicalCompound['name'],
                        'preclinicalName': preclinicalCompound['name']
                    }
                    result[inchiKey][preclinicalCompound['source']] = preclinicalCompound['findingIds']
                result[inchiKey][clinicalCompound['source']] = clinicalCompound['findingIds']
    return result


def getClinicalCompounds(api):
    ct_compounds = api.ClinicalTrials().getAllCompounds();
    for ct_compound in ct_compounds:
        ct_compound['source'] = 'ClinicalTrials'
    ml_compounds = api.Medline().getAllCompounds();
    for ml_compound in ml_compounds:
        ml_compound['source'] = 'Medline'
    fa_compounds = api.Faers().getAllCompounds();
    for fa_compound in fa_compounds:
        fa_compound['source'] = 'Faers'
    dm_compounds = api.DailyMed().getAllCompounds();
    for dm_compound in dm_compounds:
        dm_compound['source'] = 'DailyMed'

    return ct_compounds + ml_compounds + fa_compounds + dm_compounds


def getPreclinicalCompounds(api):
    et_compounds = api.eToxSys().getAllCompounds()
    for et_compound in et_compounds:
        et_compound['source'] = 'eToxSys'
    return et_compounds


def getFindingsByIds(api, service, findingIds):
    result = []
    record_count = 0

    query = {
        "filter": {
            "criteria": [
                [
                    {
                        "field": {
                            "dataClassKey": "FINDING",
                            "name": "id"
                        },
                        "primitiveType": "Integer",
                        "comparisonOperator": "IN",
                        "values": None
                    },
                ]
            ]
        },
        "selectedFields": [
            {
                "dataClassKey": "FINDING",
                "names": [
                    "id",
                    "specimenOrgan", "specimenOrganCode", "specimenOrganVocabulary",
                    "findingIdentifier", "finding", "findingCode", "findingVocabulary", "findingType",
                    "severity", "observation", "frequency",
                    "dose", "doseUnit",
                    "timepoint", "timepointUnit",
                    "treatmentRelated",
                    "compoundId",
                    "studyId",
                    "createdDate", "modifiedDate", "sex"
                ]
            }
        ],
        "offset": 0,
        "limit": 500
    }

    for offset in range(0, len(findingIds), 500):
        query['filter']['criteria'][0][0]['values'] = [{'value': findingId} for findingId in findingIds[offset:offset+500]]
        r = requests.post(service.endpoint + 'query', verify=False, headers={"Authorization": f"Bearer {api.get_token()}"}, json=query, timeout=None)

        if r.status_code == 200:
            response = json.loads(r.text)
            for record in response['resultData']['data']:
                record['FINDING']['source'] = response['origin']
                result.append(record['FINDING'])
        elif r.status_code == 401:
            api.reconnect()
            continue

    return result

