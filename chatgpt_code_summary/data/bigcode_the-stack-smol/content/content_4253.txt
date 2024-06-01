import os
import re
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from zipfile import ZipFile

import helpers.config as config
from helpers.logger import Logger


class Updater:
    __instance = None

    @staticmethod
    def Get():
        if Updater.__instance is None:
            return Updater()
        return Updater.__instance

    def __init__(self):
        if Updater.__instance is not None:
            return
        else:
            self.log = Logger("pyLaunch.Frontend.Updater", "frontend.log")
            self.DeleteFolders = ["src"]
            self.UpdateFolder = "updatefiles"

    def Automatic(self) -> bool:
        if not self.CheckConnection():
            return False
        UpdateAvailable = self.CheckVersions()
        if UpdateAvailable:
            print(f"An update is available! [v{'.'.join(self.Versions[1])}]")
            if not 'n' in input(f"Would you like to update from [{'.'.join(self.Versions[0])}]? (Y/n) > "):
                if self.DownloadUpdate():
                    return self.InstallUpdate()
        return False

    def CheckConnection(self) -> str:
        if config.CONFIGURATION['Update']['SkipCheck']:
            return "Skipping update check"
        try:
            urllib.request.urlopen('http://google.com')
            return True
        except Exception as e:
            return "Unable to connect to the internet" # Unable to connect to the internet

    def DownloadUpdate(self) -> bool:
        response = None
        try:
            response = urllib.request.urlopen(f"https://api.github.com/repos/{config.CONFIGURATION['Update']['Organization']}/{config.CONFIGURATION['Update']['Repository']}/zipball/{config.CONFIGURATION['Update']['Branch']}")
        except urllib.error.HTTPError as e:
            print(f"Unable to download update from GitHub: {e}")
            input("Press enter to continue...")
            return False

        if not os.path.exists(f"{config.PATH_ROOT}{os.sep}{self.UpdateFolder}"):
            os.mkdir(f"{config.PATH_ROOT}{os.sep}{self.UpdateFolder}")
        with open(f"{config.PATH_ROOT}{os.sep}{self.UpdateFolder}{os.sep}gh_download.zip", "wb") as f:
            f.write(response.read())

        # Zip is downloaded, now extract
        os.chdir(f"{config.PATH_ROOT}{os.sep}{self.UpdateFolder}")
        zipFileContent = dict()
        zipFileContentSize = 0
        with ZipFile(f"gh_download.zip", 'r') as zipFile:
            for name in zipFile.namelist():
                zipFileContent[name] = zipFile.getinfo(name).file_size
            zipFileContentSize = sum(zipFileContent.values())
            extractedContentSize = 0
            for zippedFileName, zippedFileSize in zipFileContent.items():
                UnzippedFilePath = os.path.abspath(f"{zippedFileName}")
                os.makedirs(os.path.dirname(UnzippedFilePath), exist_ok=True)
                if os.path.isfile(UnzippedFilePath):
                    zipFileContentSize -= zippedFileSize
                else:
                    zipFile.extract(zippedFileName, path="", pwd=None)
                    extractedContentSize += zippedFileSize
                try:
                    done = int(50*extractedContentSize/zipFileContentSize)
                    percentage = (extractedContentSize / zipFileContentSize) * 100
                except ZeroDivisionError:
                    done = 50
                    percentage = 100
                sys.stdout.write('\r[{}{}] {:.2f}%'.format('█' * done, '.' * (50-done), percentage))
                sys.stdout.flush()
        sys.stdout.write('\n')
        os.chdir(config.PATH_ROOT)
        return True

    def InstallUpdate(self) -> bool:
        print("Installing new version")
        for file in os.listdir(config.CONFIGURATION['Launch']['ProjectRoot']):
            if os.path.isdir(f"{config.CONFIGURATION['Launch']['ProjectRoot']}{os.sep}{file}"):
                if file in self.DeleteFolders:
                    shutil.rmtree(f"{config.CONFIGURATION['Launch']['ProjectRoot']}{os.sep}{file}")
            else: # Files
                os.remove(f"{config.CONFIGURATION['Launch']['ProjectRoot']}{os.sep}{file}")

        # Old version is deleted

        for file in os.listdir(f"{config.PATH_ROOT}{os.sep}{self.UpdateFolder}"):
            os.rename(f"{config.PATH_ROOT}{os.sep}{self.UpdateFolder}{os.sep}{file}", f"{config.CONFIGURATION['Launch']['ProjectRoot']}{os.sep}{file}")
        shutil.rmtree(f"{config.PATH_ROOT}{os.sep}{self.UpdateFolder}")
        return True

    def CheckVersions(self):
        # Sucessful return: bool
        # Unsuccessful: list[message: str, continue: bool]
        self.Versions = self._GetVersions()
        if type(self.Versions[1]) == bool:
            return self.Versions

        self.Versions[0] = self._GetVersionAsInt(self.Versions[0])
        self.Versions[1] = self._GetVersionAsInt(self.Versions[1])
        self.Difference = []
        for installed, checked in zip(self.Versions[0], self.Versions[1]):
            self.Difference.append(checked - installed)
        
        for section in self.Difference:
            if section < 0: # When working on project and updating locally
                return False
            elif section > 0:
                return True
        return False

    def _GetVersions(self) -> list:
        # Sucessful return: list[InstalledVersion: str, CheckedVersion: str]
        # Unsucessful: list[message: str, continue: bool]
        if not os.path.exists(f"{config.CONFIGURATION['Launch']['ProjectRoot']}{os.sep}{config.CONFIGURATION['Update']['VersionPath']}"):
            # This means either the configuration is incorrect, or pyLaunch isn't where it should be
            # continue is False, because the project cannot be launched
            return [f"Unable to locate installed version at {config.CONFIGURATION['Update']['VersionPath']}", False]

        InstalledVersion = None # Local Version
        CheckedVersion = None # Version on GitHub

        with open(f"{config.CONFIGURATION['Launch']['ProjectRoot']}{os.sep}{config.CONFIGURATION['Update']['VersionPath']}", "r") as f:
            lines = f.readlines()
            InstalledVersion = self._GetVersionFromStr(lines)

        try:
            response = urllib.request.urlopen(f"https://raw.githubusercontent.com/{config.CONFIGURATION['Update']['Organization']}/{config.CONFIGURATION['Update']['Repository']}/{config.CONFIGURATION['Update']['Branch']}{config.CONFIGURATION['Update']['VersionPath']}")
            content = response.read().decode("UTF-8").split("\n")
            CheckedVersion = self._GetVersionFromStr(content)
        except urllib.error.HTTPError as e:
            # The Project URL is invalid (cannot find Org/Repo/Branch/VersionPath) or,
            # raw.githubusercontent is down, continue is True, the project can still be launched
            return ["Project URL does not exist or githubusercontent is down", True] # URL doesn't exist or something went wrong

        if CheckedVersion is None:
            # Some other error, just to be safe.
            return ["Unable to get current version from GitHub", True]
        return [InstalledVersion, CheckedVersion]

    def _GetVersionFromStr(self, lines: str) -> str:
        ver = None
        for line in lines:
            line = line.strip()
            if config.CONFIGURATION['Update']['Find'] in line:
                ver = line[len(config.CONFIGURATION['Update']['Find']):].strip('"')
        match = re.match(r"\d+\.\d+\.\d+", ver) # > #.#.#

        if match:
            return ver[match.start():match.end()]
        return None

    def _GetVersionAsInt(self, version: str) -> list:
        version = version.split(".")
        intVer = []
        for section in version:
            if section.isalnum():
                newSection = ""
                for char in section:
                    if char.isnumeric():
                        newSection += char
                section = newSection
            intVer.append(int(section))
        return intVer
