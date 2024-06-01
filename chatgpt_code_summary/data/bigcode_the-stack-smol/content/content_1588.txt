import sys

from subprocess import Popen

import cx_Oracle

root_directory = sys.argv[1]

def main(directory):
    public_projects = get_public_project_accessions()
    for project_accession in public_projects:
        Popen(['./runAnnotator.sh', directory, str(project_accession)])


# get all the project references from pride archive
def get_public_project_accessions():
    accessions = list()
    archive_cursor = connect_archive()

    archive_cursor.execute(
        "select accession from project where (submission_type='PRIDE' or submission_type='COMPLETE') and is_public = 1")
    projects = archive_cursor.fetchall()

    for project in projects:
        accessions.append(project[0])

    archive_cursor.close()
    return accessions


# connect to pride archive database
def connect_archive():
    # connect to archive database
    archive_db = cx_Oracle.connect(
        "${pride.repo.db.user}/${pride.repo.db.password}@(DESCRIPTION=(ADDRESS=(PROTOCOL=tcp)(HOST=ora-vm-032.ebi.ac.uk)(PORT=1531))(CONNECT_DATA=(SERVICE_NAME=PRIDEPRO)))")

    # Create an cursor object for archive database
    return archive_db.cursor()


if __name__ == '__main__':
    main(root_directory)