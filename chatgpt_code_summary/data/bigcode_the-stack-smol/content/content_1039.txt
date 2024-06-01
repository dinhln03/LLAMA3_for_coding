import pymysql.cursors
import ldap


def get_domain_name():
    """
    Returns the domain name of the current configuration from a config file
    
    Returns
    -------
    string
        the domain name
    """
    with open("/var/www/logic_webapp/webapp_config") as file:
        line = file.readline()
        domain = line.split("=")[1].rstrip()  # Take right hand side of = and remove \n
        return domain


def get_db_password():
    with open("/var/www/logic_webapp/webapp_config") as file:
        line = file.readlines()[1]
        password = line.split("=")[
            1
        ].rstrip()  # Take right hand side of = and remove \n
        return password


def create_slurm_db_connection(host, port, user, password, db):
    """
    Creates the connection to the database (MySQL) so it can be queried
    
    Parameters
    ----------
    host : string
        hostname on which is located the DB
    port : integer
        port on which the connection is to be established
    user : string
        user name with which the connection is to be established
    password : string
        password of the user on the database (of the user `user`)
    db : string
        name of the database which will be queried

    Returns
    -------
    PyMySQL Connection object
    """

    connection = pymysql.connect(
        host=host, port=port, user=user, password=password, db=db,
    )
    print("[+] Slurm accounting DB connection is up! [+]")
    return connection


def create_ldap_connection(host):
    """
    Creates an LDAP connection object with a given hostname

    Parameters
    ----------
    host : hostname with the LDAP database in the form of (ldap://host)

    Returns
    -------
    LDAP connection object
    """
    connection = ldap.initialize(host)
    connection.set_option(ldap.OPT_REFERRALS, 0)
    connection.simple_bind_s()
    return connection
