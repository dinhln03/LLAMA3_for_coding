
import os
import sys
path = os.getcwd() 
package_path = (os.path.abspath(os.path.join(path, os.pardir))).replace('\\', '/')+'/'
sys.path.insert(1, package_path)
from config.config import *


##############################################Scrape-1###################################################
def contains(text , subtext):
    if subtext in text:
        return True
    return False


def get_scrape_url(url):
    encoding = "html.parser"
    resp = requests.get(url)
    http_encoding = resp.encoding if 'charset' in resp.headers.get('content-type', '').lower() else None
    html_encoding = EncodingDetector.find_declared_encoding(resp.content, is_html=True)
    encoding = html_encoding or http_encoding
    soup = BeautifulSoup(resp.content, from_encoding=encoding)

    for link in soup.find_all('a', href=True):
        scrape_url = str(link['href'])
        if(contains(scrape_url , "s3.amazonaws.com") and contains(scrape_url , ".zip")):
            break
    file_name = scrape_url.split("/Kickstarter/")[1]
    return scrape_url, file_name

def download(scrape_url , output_directory):
    try:
        wget.download(scrape_url, out=output_directory)
    except:
        raise Exception("Failed in downloading the data file")
    return output_directory


def unzip_data(input_file_path , output_directory):
    try:
        with zipfile.ZipFile(input_file_path, 'r') as zip_ref:
            zip_ref.extractall(output_directory)
    except Exception as e:
        raise Exception("Failed to unzip the data folder !+....{}",format(e))
    os.remove(input_file_path)
    return True

###################################scrape-1ends############################################################
