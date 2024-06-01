'''

@author: xiayuanhuang

'''

import csv
import matchECtoDemog
import decisionTreeV7
import family_treeV4
import inference
import combine_new_ped

def combine(addressFile, nameFile, demoFile, accountFile, outputFile, patientFile, ecFile, familyTreeOutput):
    reader_add = csv.reader(open(addressFile, 'r'), delimiter = ',')
    h_add = next(reader_add)
    exp_header_add = ['study_id', 'street_1', 'street_2', 'city', 'state', 'zip', 'from_year', 'thru_year']
    if not h_add == exp_header_add:
        raise Exception("Address file (%s) doesn't have the header expected: %s" % (addressFile, exp_header_add))
    
    reader_name = csv.reader(open(nameFile, 'r'), delimiter = ',')
    h_name = next(reader_name)
    exp_header_name = ['study_id', 'last_name_id', 'first_name_id', 'middle_name_id', 'from_year', 'thru_year']
    if not h_name == exp_header_name:
        raise Exception("Name file (%s) doesn't have the header expected: %s" % (nameFile, exp_header_name))
    
    reader_demo = csv.reader(open(demoFile, 'r'), delimiter = ',')
    h_demo = next(reader_demo)
    exp_header_demo = ['study_id', 'GENDER_CODE', 'birth_year', 'deceased_year', 'PHONE_NUM_id', 'from_year', 'thru_year']
    if not h_demo == exp_header_demo:
        raise Exception("Demographic data file (%s) doesn't have the header expected: %s" % (demoFile, exp_header_demo))

    reader_acc = csv.reader(open(accountFile, 'r'), delimiter = ',')
    h_acc = next(reader_acc)
    exp_header_acc = ['study_id', 'ACCT_NUM_id', 'from_year', 'thru_year']
    if not h_acc == exp_header_acc:
        raise Exception("Account file (%s) doesn't have the header expected: %s" % (accountFile, exp_header_acc))

    reader_p = csv.reader(open(patientFile, 'r'), delimiter = ',')
    h_p = next(reader_p)
    exp_header_p = ['PatientID', 'FirstName', 'LastName', 'Sex', 'PhoneNumber', 'Zipcode', 'birth_year', 'deceased_year']
    if not h_p == exp_header_p:
        raise Exception("Patient data file (%s) doesn't have the header expected: %s" % (patientFile, exp_header_p))

    reader_ec = csv.reader(open(ecFile, 'r'), delimiter = ',')
    h_ec = next(reader_ec)
    exp_header_ec = ['PatientID', 'EC_FirstName', 'EC_LastName', 'EC_PhoneNumber', 'EC_Zipcode', 'EC_Relationship']
    if not h_ec == exp_header_ec:
        raise Exception("Emergency contact data file (%s) doesn't have the header expected: %s" % (ecFile, exp_header_ec))


    args = input("Enter one PED file if any:")
    if args != '':
        ped = args.strip().split(' ')[0]
        reader_ped = csv.reader(open(ped, 'r'), delimiter = ',')
        h_ped = next(reader_ped)
        exp_header_ped = ['familyID', 'family_member', 'study_ID', 'StudyID_MATERNAL', 'StudyID_PATERNAL', 'Sex']
        if not h_ped == exp_header_ped:
            raise Exception("PED data file (%s) doesn't have the header expected: %s" % (ped, exp_header_ped))

        ### run combined algorithm
        ### riftehr

        mt = matchECtoDemog.matches(patientFile, ecFile)
        mt.assignFamily('riftehr_pedigree.csv')

        ### fppa

        newDT = decisionTreeV7.DT(addressFile, nameFile, demoFile, accountFile)
        newDT.predict()
        newDT.writeToFile(outputFile)

        newFamilyTree = family_treeV4.familyTree(newDT)
        newFamilyTree.filter(outputFile)
        newFamilyTree.buildTree()

        #newFamilyTree.connected('fppa_pedigree.csv')

        new_infer = inference.matches(mt.qc_matches, mt.sex, newFamilyTree.edges, newFamilyTree.gender, familyTreeOutput)
        #new_infer.assignFamilies(familyTreeOutput)

        comb = combine_new_ped.matches(ped, new_infer.ec, new_infer.sex, new_infer.fppa_pair, new_infer.p_c_gender, new_infer.famOut)

        
    else:
        ### run combined algorithm
        ### riftehr

        mt = matchECtoDemog.matches(patientFile, ecFile)
        mt.assignFamily('riftehr_pedigree.csv')

        ### fppa

        newDT = decisionTreeV7.DT(addressFile, nameFile, demoFile, accountFile)
        newDT.predict()
        newDT.writeToFile(outputFile)

        newFamilyTree = family_treeV4.familyTree(newDT)
        newFamilyTree.filter(outputFile)
        newFamilyTree.buildTree()

        #newFamilyTree.connected('fppa_pedigree.csv')

        new_infer = inference.matches(mt.qc_matches, mt.sex, newFamilyTree.edges, newFamilyTree.gender, familyTreeOutput)
        new_infer.assignFamilies()



'''
    ### run combined algorithm
    ### riftehr

    mt = matchECtoDemog.matches(patientFile, ecFile, 'riftehr_pedigree.csv')

    ### fppa

    newDT = decisionTreeV7.DT(addressFile, nameFile, demoFile, accountFile)
    newDT.predict()
    newDT.writeToFile(outputFile)

    newFamilyTree = family_treeV4.familyTree(newDT)
    newFamilyTree.filter(outputFile)
    newFamilyTree.buildTree()

    #newFamilyTree.connected('fppa_pedigree.csv')

    new_infer = inference.matches(mt.qc_matches, mt.sex, newFamilyTree.edges, newFamilyTree.gender, familyTreeOutput)

'''