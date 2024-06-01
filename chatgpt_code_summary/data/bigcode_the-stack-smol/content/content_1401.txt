#! /usr/bin/env python
# coding utf-8


import sys
from sys import exit
import os
import socket
import requests
import smtplib
import ssl
import dns.resolver


""" Python script to monitor list of url (https/http/ns/mx)
and send mail if down"""

__author__ = "Benjamin Kittler"
__copyright__ = "Copyright 2021, KITTLER"
__credits__ = ["Benjamin Kittler"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Benjamin Kittler"
__email__ = "kittler @T. gmail. com"
__status__ = "integration"

"""
############################################################
# Please complete these variable before the first launch #
############################################################
"""
# mail provider : TO BE MODIFIED
smtp_address = 'smtp.gmail.com'
smtp_port = 465

# email address and password : TO BE MODIFIED
email_address = 'EMAIL@EMAIL.COM'
email_password = 'PASSWORD'

""" Python script to monitor list of url (https/http/ns/mx)
and send mail if down"""


def check(file_to_check, testmode, debug):
    """
    Function open file, read each line and complete a dictionnary
    For each entry, launch check url : http/https or launch resolution then ping for MX/NS entry
    If one url not respond, launch email to alert   

    Parameters
    ----------
    file_to_check : string
        This is the name of the fillethat contain list of url must be checked
        and mail  for alert
    testmode : string
        This value is 0 by defaut and is to 1 if user launchscript on test mode:
        print enabled and no mail send
    debug : string
        This value is 0 by defaut and is to 1 if user launchscript on debug mode:
        more print enabled and no mail send

    Returns
    -------
    None.

    """
    try:
        file = open(file_to_check, "r")
    except:
        exit('open file failed')

    # lines contain all line of file
    lines = file.readlines()
    # close the file after read all lines
    file.close()
    # create dict of url
    url_dict = {}
    # add each element on dict
    for line in lines:
        # clean end of line contain \n
        line = line.replace("\n", "")
        # clean line contain multiple space
        line = line.replace(" ", "\t")
        # clean line contain multiple \t
        line = line.replace("\t\t\t", "\t")
        line = line.replace("\t\t", "\t")
        # clean line contain http:// or https://
        line = line.replace("http://", "")
        line = line.replace("https://", "")
        element = line.split("\t")
        cle = element[0]
        data = element[1]
        url_dict[cle] = data

    if debug == 1:
        print("Url dict : \n", url_dict)

    if testmode == 1:
        print("Check :")
    for url, mail in url_dict.items():
        # check http or https entry
        if "ns://" not in url and "mx://" not in url and "ping://" not in url:
            availability = str(request_url(url))
            # import pdb; pdb.set_trace()
            if (availability == ("200") or (availability == "301")
            or (availability == "302")):
                request_url_result = "UP"
            else:
                request_url_result = "DOWN"
            if testmode == 1:
                print("url : ", url, "  -> mail : ", mail, 
                      " Result :", request_url_result)
            else:
                if request_url_result == "DOWN":
                    # print("mail :", mail)
                    alert_mail(mail, request_url_result, url)
        # check ns entry
        elif "ns://" in url:
            request_url_result = ping_name(url, "NS")
            if testmode == 1:
                print("url : ", url, "  -> mail : ", mail,
                      " Result NS :", request_url_result)
            else:
                if request_url_result == "DOWN":
                    # print("mail :", mail)
                    alert_mail(mail, request_url_result, url)
        # check mx entry
        elif "mx://" in url:
            request_url_result = ping_name(url, "MX")
            if testmode == 1:
                print("url : ", url, "  -> mail : ", mail,
                      " Result MX :", request_url_result)
            else:
                if request_url_result == "DOWN":
                    # print("mail :", mail)
                    alert_mail(mail, request_url_result, url)
        # check ping entry
        elif "ping://" in url:
            url = url.replace("ping://", "")
            request_url_result = ping_ip(url)
            if testmode == 1:
                print("url : ", url, "  -> mail : ", mail,
                      " Result Ping :", request_url_result)
            else:
                if request_url_result == "DOWN":
                    # print("mail :", mail)
                    alert_mail(mail, request_url_result, url)
        # ignore entry
        else:
            if testmode == 1:
                print("url : ", url, "  -> mail : ", mail, "ignored")
    exit()


def request_url(url):
    """
    Function to send https or http request to this url and return code result.

    Parameters
    ----------
    url : string
        This variable contain url must be checked

    Returns
    -------
    status_code : int
        Code result

    """
    try:
        url = "https://" + format(url)
        response = requests.head(url, allow_redirects=True, timeout=10)
    except:
        try:
            url = "http://" + format(url)
            response = requests.head(url, allow_redirects=True, timeout=10)
        except:
            return "404"
            # print("Request failed")
    if response.status_code:
        return response.status_code
    else:
        return "404"

def ping_name(name, dns_type):
    """
    Function to resolve name and ping this host.
    print the result of ping

    Parameters
    ----------
    name : string
        This variable contain the name (host) must be checked
    dns_type : string
        This variable contain the DNS type : A, NS, MX 

    Returns
    -------
    status : String
        Status result : UP or DOWN

    """
    # clean name host
    name = name.replace("ns://", "")
    name = name.replace("mx://", "")
    
    # make resolution
    if dns_type == "A":
        try:
            addr1 = socket.gethostbyname_ex(name)
            print("Resolution -> {}".format(addr1[2]))
            name = addr1[2]
        except:
            print("Resolution failed")

    # make resolution
    if dns_type == "MX":
        try:

            answers = dns.resolver.resolve(name, 'MX')
            for rdata in answers:
                # import pdb; pdb.set_trace()
                #print('Mail exchange:',rdata.exchange)
                addr1 = socket.gethostbyname_ex(str(rdata.exchange))
                #print("Resolution -> {}".format(addr1[2]))
                name = addr1[2]
                if ping_ip(name) == "UP":
                    return "UP"
            return ping_ip(name)
        except:
            print("Resolution failed")
            return "DOWN"

    # make resolution
    if dns_type == "NS":
        try:

            answers = dns.resolver.resolve(name, 'NS')
            for rdata in answers:
                #import pdb; pdb.set_trace()
                #print('Mail exchange:',rdata.exchange)
                addr1 = socket.gethostbyname_ex(str(rdata.target))
                #print("Resolution -> {}".format(addr1[2]))
                name = addr1[2]
                for srv in name:
                    if ping_ip(srv) == "UP":
                        return "UP"
            return ping_ip(name)
        except:
            print("Resolution failed")
            return "DOWN"

def ping_ip(name):
    """
    Function to ping name.
    return the result of ping

    Parameters
    ----------
    name : string
        This variable is IP address

    Returns
    -------
    status : String
        Status result : UP or DOWN

    """
    try:
        # import pdb; pdb.set_trace()
        name = str(name).strip('[]')
        name = str(name).strip("''")
        hostname = format(name)
        response = os.system("ping -c 1 " + hostname + " > /dev/null 2>&1")
        # import pdb; pdb.set_trace()
        if response == 0:
            return "UP"
            # print("Response ping : OK")
        else:
            return "DOWN"
            # print("Response ping : KO")
    except requests.ConnectionError:
        return "DOWN"
        # print("Response ping : failed to connect")
    return "DOWN"

def alert_mail(email_receiver, service_status, url):
    """
    Function to send email Alert

    Parameters
    ----------
    email_receiver : string
        destination email for alert
    service_status : string
        service status
    url : string
        url concertned by alert

    Returns
    -------
    None.

    """
    # create subject
    service_status = "Subject:{}\n\n".format(service_status) + "Server :{} \n".format(url)
    
    # create connexion
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_address, smtp_port, context=context) as server:
      # account connexion
      server.login(email_address, email_password)
      # sending mail
      server.sendmail(email_address, email_receiver, service_status)


def main(argv, testmode, debug):
    """
    Print the fileopened and lauchn the check of file with testmode / debug value
    
    Parameters
    ----------
    file_to_check : string
        This is the name of the fillethat contain list of url must be checked
        and mail  for alert
    testmode : string
        This value is 0 by defaut and is to 1 if user launchscript on test mode:
        print enabled and no mail send
    debug : string
        This value is 0 by defaut and is to 1 if user launchscript on debug mode:
        more print enabled and no mail send

    Returns
    -------
    None.

    """
    # print argument for verification
    if testmode == 1:
        print("Import file: {}".format(argv[0]))
    file = str(argv[0])
    # launch check file entry
    check(file, testmode, debug)
    
if __name__ == "__main__":
    """
    Get arguments from command line and fixe value :
    testmode : 
        This value is 0 by defaut and is to 1 if user launchscript on test mode:
        print enabled and no mail send
    debug : 
        This value is 0 by defaut and is to 1 if user launchscript on debug mode:
        more print enabled and no mail send
    
    call main with  arguments

    """
    # pretrieve argument, seach test mode and launch main
    if "-t" in sys.argv:
        testmode = 1
        debug = 0
    elif "--test" in sys.argv:
        testmode = 1
        debug = 0
    elif "--debug" in sys.argv:
        testmode = 1
        debug = 1
    else:
        testmode = 0
        debug = 0
    
    matching = [cmd for cmd in sys.argv if ".txt" in cmd]
    main(matching, testmode, debug)
