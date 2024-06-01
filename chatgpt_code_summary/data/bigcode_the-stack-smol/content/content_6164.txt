from PIL import Image
import gspread
import hashlib

from googleapiclient.errors import HttpError
from oauth2client.service_account import ServiceAccountCredentials
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload
import io
import json


def has_transparency(image: Image) -> bool:
    if image.mode == "P":
        transparent = image.info.get("transparency", -1)
        for _, index in image.getcolors():
            if index == transparent:
                return True
    elif image.mode == "RGBA":
        extrema = image.getextrema()
        if extrema[3][0] < 255:
            return True

    return False


# from https://stackoverflow.com/questions/61201141/how-can-i-crop-an-image-with-11-aspect-ratio-using-pillow-in-python
def crop_image(image):
    width, height = image.size
    if width == height:
        return image
    offset = int(abs(height - width) / 2)
    if width > height:
        image = image.crop([offset, 0, width - offset, height])
    else:
        image = image.crop([0, offset, width, height - offset])
    return image


def resize_image(image):
    width, height = image.size
    new_width = 256
    if width < new_width:
        return image
    new_height = new_width * height / width
    return image.resize((new_width, int(new_height)), Image.ANTIALIAS)


def LongestBio(section: list) -> int:
    longestBio = 0
    for items in section:
        if longestBio < len(items["description"]):
            longestBio = len(items["description"])
    return longestBio


def sortBioLength(bioList: list) -> list:
    holdingList = []
    workingList = bioList
    for x in range(len(workingList)):
        highest = LongestBio(workingList)
        itemIndex = -1
        for items in workingList:
            itemIndex = itemIndex + 1
            if len(items["description"]) == highest:
                holdingList.append(items)
                break
        if itemIndex != -1:
            del workingList[itemIndex]
    return holdingList

# Deals with getting all the auth setup for the connecting to GSheet
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
# Needs to link to auth file given by google dev dashboard
creds = ServiceAccountCredentials.from_json_keyfile_name("secrets/googleAuth.json", scope)
client = gspread.authorize(creds)
sheet = client.open("Low Ink Staff Bio Form (Responses)")  # Name of the google sheet file
worksheet = sheet.worksheet("input")  # name of the sheet in question
worksheetData = worksheet.get_all_records()

# This is the auth scope for Google Drive API
creds = None
if os.path.exists('secrets/token.pickle'):
    with open('secrets/token.pickle', 'rb') as token:
        creds = pickle.load(token)
# If there are no (valid) credentials available, let the user log in.
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            'secrets/credentials.json', ['https://www.googleapis.com/auth/drive'])
        creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open('secrets/token.pickle', 'wb') as token:
        pickle.dump(creds, token)

service = build('drive', 'v3', credentials=creds)

commentator = []
former = []
headTO = []
orgHead = []
production = []
staff = []
artists = []

for lines in worksheetData:
    output = {}
    if lines["isStaff"] in ["Yes (Staff temp invite)", "Yes", "Yes (Staff, temp, invite)"]:
        print("Outputting for: {}".format(lines["name"]))
        staffID = hashlib.md5(lines["name"].encode("utf-8")).hexdigest()

        # Obtains image from google drive
        imageID = (lines["image"].split("?id="))[1]  # get the G Drive file ID from share
        request = service.files().get_media(fileId=imageID)
        fh = io.FileIO("holding/{}.png".format(staffID), "wb")  # states where the file saves to
        # Downloads file
        downloader = MediaIoBaseDownload(fh, request)
        output_file_format = 'png'
        done = False
        while done is False:
            try:
                status, done = downloader.next_chunk()
            except HttpError as e:
                print("Could not output image. Please provide file '{}' manually.".format("output/images/{}.png").format(staffID))
                print("Error message: '{}'".format(e.error_details[0]["message"]))
                break

        if done is True:
            # Crops image to be 1:1
            staff_image = crop_image(Image.open("holding/{}.png".format(staffID)))
            staff_image = resize_image(staff_image)
            if has_transparency(staff_image):
                staff_image.convert("P", palette=Image.ADAPTIVE).save("output/images/{}.png".format(staffID))
            else:
                staff_image.convert("RGB", palette=Image.ADAPTIVE).save("output/images/{}.jpg".format(staffID))
                output_file_format = 'jpg'
            staff_image.close()

        output = {
            "title": lines["name"],
            "description": (lines["bio"].replace("\n", "")).replace("\r", " "),
            "imagePath": "images/Staff/{}.{}".format(staffID, output_file_format),
            "twitter": lines["twitter"],
            "credit": lines["credits"]
        }

        # Save bio to right list
        if lines["header"] == "General Staff":
            staff.append(output)
        elif lines["header"] == "Commentators":
            commentator.append(output)
        elif lines["header"] == "Head TO":
            headTO.append(output)
        elif lines["header"] == "Production & Development":
            production.append(output)
        elif lines["header"] == "Org Head":
            orgHead.append(output)
        elif lines["header"] == "Temp staff":
            staff.append(output)
        elif lines["header"] == "Former staff":
            former.append(output)
        elif lines["header"] == "Guest Staff":
            staff.append(output)
        elif lines["header"] == "Artist":
            artists.append(output)

staffFile = [
    {"elemClassName": "staff-layout-grid",
     "contents": sortBioLength(staff)},
    {"elemClassName": "org-head-grid",
     "contents": sortBioLength(orgHead)},
    {"elemClassName": "head-TO-grid",
     "contents": sortBioLength(headTO)},
    {"elemClassName": "production-grid",
     "contents": sortBioLength(production)},
    {"elemClassName": "commentator-grid",
     "contents": sortBioLength(commentator)},
    {"elemClassName": "former-staff-grid",
     "contents": sortBioLength(former)},
    {"elemClassName": "artists-staff-grid",
     "contents": sortBioLength(artists)}
]

with open('output/staff.json', 'w') as file:
    json.dump(staffFile, file)
