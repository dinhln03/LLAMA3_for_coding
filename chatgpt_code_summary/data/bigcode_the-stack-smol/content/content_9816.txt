import os, paramiko, time, schedule, smtplib, ssl
from datetime import datetime
from email.message import EmailMessage

host='localhost'
port='5432'
user='postgres'
password='admin'
database='testdb'

#chemin de sauvegarde locale
local_dir = 'C:\\Users\\Kamla\\projets\\auto-backup-sqldb\\backup\\'
#local_dir = 'Chemin vers le dossier de la base de donnees a sauvegarder\\'

#chemin de sauvegarde distant
remote_dir = '/C:/Users/vmwin10/Documents/ftpfile/'

def job():
    print("Backup working...")
    
    filestamp = time.strftime('%Y-%m-%dT%H-%M-%S.%z')
    
    #nom pour le fichier sql qui serra genere par pg_dump
    database_remote = database+"_"+filestamp+".bak.sql"
    
    PASS="set PGPASSWORD=%s" % (password)
    #lancement de la commande mysqldump qui va faire une sauvegarde en local
    #les fichiers sont sauvegarder dans le respertoire 'backup'
    os.system("(cd backup) && ("+PASS+") && (pg_dump -h %s -p %s -U %s -f %s -C -d %s)" % (host, port, user, database_remote, database))
    
    print("Database dumped to "+database_remote)
    
    # debut du SFTP
    ssh_client=paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    #on se connecte a la machine dans laquelle serra sauvegarde le le fichier backup
    ssh_client.connect(hostname='192.168.126.2',username='vmwin10',password='vmwin10')

    ftp_client=ssh_client.open_sftp()
    
    #envoie du fichier local vers le remote
    ftp_client.put(local_dir+database_remote,remote_dir+database_remote)
    ftp_client.close()
    print("Successfull Backup")
    
    # A chaque backup un email est envoye
    msg = EmailMessage()
    msg.set_content("Un backup vient d'etre effectue")
    msg["Subject"] = "Email de Backup"
    msg["From"] = "ksb.cmr@gmail.com"
    msg["To"] = "test@mail.com"
    context=ssl.create_default_context()
    with smtplib.SMTP("smtp.gmail.com", port=587) as smtp:
        smtp.starttls(context=context)
        smtp.login(msg["From"], "password")
        smtp.send_message(msg) 
    
# le backup se fait chaque 1h

schedule.every(3).seconds.do(job)
#schedule.every(15).minutes.do(job)
#schedule.every().hour.do(job)
#schedule.every().day.at("10:30").do(job)
#schedule.every(10).to(10).minutes.do(job)
#schedule.every().monday.do(job)
#schedule.every().wednesday.at("15:00").do(job)
#schedule.every().minute.at(":15").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
