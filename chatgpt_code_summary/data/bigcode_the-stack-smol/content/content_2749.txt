#!/usr/bin/env python

import os
import xmltodict  # sudo easy_install xmltodict
import subprocess
import zipfile


class PackAndroid(object):

    def __init__(self, root, project_folder, project, input_apk, destination, keystore, keystore_alias, apk_name=None, zipalign=None, jarsigner=None, configuration='Release', keystore_password=None):
        self.name = project_folder
        self.proj_folder = project_folder
        self.project = project
        self.input_apk = input_apk
        self.destination = os.path.expanduser(destination)
        self.configuration = configuration

        self.keystore = keystore
        self.keystore_alias = keystore_alias
        self.keystore_password = keystore_password

        # Name of the final apk
        self.apk_name = apk_name
        if self.apk_name is None and self.keystore_alias is not None:
            self.apk_name = self.keystore_alias.lower()
        if self.apk_name is None:
            projf = os.path.basename(project)
            self.apk_name = projf.replace('.csproj', '')
        self.final_apk = os.path.join(self.destination, "%s-" % self.apk_name)
        self.signed_apk = os.path.join(self.destination, "%s-signed.apk" % self.apk_name)

        self.zipalign = zipalign
        if self.zipalign is None:
            self.zipalign = '/usr/bin/zipalign'

        self.jarsigner = jarsigner
        if self.jarsigner is None:
            self.jarsigner = "/usr/bin/jarsigner"

        self.keystore = os.path.join(root, self.keystore)
        self.project = os.path.join(root, self.project)
        self.proj_folder = os.path.join(root, self.proj_folder)
        self.input_apk = os.path.join(self.proj_folder, self.input_apk)

        if not os.path.exists(self.keystore):
            exit("Failed to locate keystore - " + self.keystore)

        if not os.path.exists(self.zipalign):
            exit("Failed to locate zipalign - " + self.zipalign)

        if not os.path.exists(self.jarsigner):
            exit("Failed to locate jarsigner - " + self.jarsigner)

    def clean(self):
        bin_folder = os.path.join(self.proj_folder, 'bin')
        obj_folder = os.path.join(self.proj_folder, 'obj')
        if os.path.exists(bin_folder):
            print 'Clearing away ' + bin_folder
            os.system('rm -fdr ' + bin_folder)
        if os.path.exists(obj_folder):
            print 'Clearing away ' + obj_folder
            os.system('rm -fdr ' + obj_folder)

    def get_manifest_dictionary(self):
        manifest = os.path.join(self.proj_folder, 'Properties/AndroidManifest.xml')
        if not os.path.exists(manifest):
            exit("Failed to locate AndroidManifest.xml - " + manifest)
        f = file(manifest)
        xml = f.read()
        f.close()
        doc = xmltodict.parse(xml)
        return doc

    def get_build_number(self):
        doc = self.get_manifest_dictionary()
        return doc['manifest']['@android:versionCode']

    def get_version_number(self):
        doc = self.get_manifest_dictionary()
        return doc['manifest']['@android:versionName']

    def set_build_number(self, build_num):
        doc = self.get_manifest_dictionary()
        doc['manifest']['@android:versionCode'] = build_num
        xml = xmltodict.unparse(doc, pretty=True)
        manifest = os.path.join(self.proj_folder, 'Properties/AndroidManifest.xml')
        if not os.path.exists(manifest):
            exit("Failed to locate AndroidManifest.xml - " + manifest)
        f = file(manifest, 'w')
        f.write(xml)
        f.close()

    def increment_build_number(self):
        build_number = self.get_build_number()
        if build_number is None:
            build_number = "1"
        else:
            build_number = str(int(build_number)+1)
        self.set_build_number(build_number)

    def decrement_build_number(self):
        build_number = self.get_build_number()
        if build_number is None:
            build_number = "1"
        else:
            build_number = str(int(build_number)-1)
        self.set_build_number(build_number)

    def set_version_number(self, version):
        doc = self.get_manifest_dictionary()
        doc['manifest']['@android:versionName'] = version
        xml = xmltodict.unparse(doc, pretty=True)
        manifest = os.path.join(self.proj_folder, 'Properties/AndroidManifest.xml')
        if not os.path.exists(manifest):
            exit("Failed to locate AndroidManifest.xml - " + manifest)
        f = file(manifest, 'w')
        f.write(xml)
        f.close()

    def build(self):
        cmd_update = "msbuild %s /t:UpdateAndroidResources /p:Configuration=%s" % (self.project, self.configuration)
        os.system(cmd_update)

        cmd = "msbuild %s /t:SignAndroidPackage /p:Configuration=%s" % (self.project, self.configuration)
        os.system(cmd)
        if not os.path.exists(self.input_apk):
            exit("Failed to build raw apk, i.e. its missing - " + self.input_apk)

    @staticmethod
    def convert_windows_path(any_path):

       chars = []

       for i in range(len(any_path)):
          char = any_path[i]
          if char == '\\':
              chars.append('/')
          else:
              chars.append(char)
       return ''.join(chars)

    @staticmethod
    def update_solution_resources(solution,configuration):
        if not os.path.exists(solution):
            exit("Failed to locate %s - " % os.path.basename(solution))
        f = file(solution)
        sln = f.read()
        f.close()
        projects = []
        lines = sln.split('\n')        
        for line in lines:
            if line.startswith("Project("):
                start = line.find(",")
                rest = line[start+3:len(line)]
                end = rest.find(",")
                projects.append(os.path.abspath(os.path.join(os.path.dirname(solution),PackAndroid.convert_windows_path(rest[0:end-1]))))
        # print projects
        for project in projects:        
            cmd_update = "msbuild %s /t:UpdateAndroidResources /p:Configuration=%s" % (project, configuration)
            os.system(cmd_update)

    def sign(self):
        sign_cmd = [self.jarsigner, "-verbose", "-sigalg", "MD5withRSA", "-digestalg", "SHA1", "-keystore", self.keystore]
        if not self.keystore_password is None:
            sign_cmd.extend(["-storepass",self.keystore_password])
        sign_cmd.extend(["-signedjar", self.signed_apk, self.input_apk, self.keystore_alias])
        subprocess.call(sign_cmd)
        subprocess.call([self.zipalign, "-f", "-v", "4", self.signed_apk, self.final_apk])
        if os.path.exists(self.final_apk):
            if os.path.exists(self.signed_apk):
                os.system('rm ' + self.signed_apk)

    def update_version(self):
        build_number = self.get_build_number()
        print build_number
        q = raw_input("Would you like to increment the build number for %s? y/n\n> " % self.apk_name)
        if q == "y":
            build_number = str(int(build_number)+1)
            self.set_build_number(build_number)

        version_number = self.get_version_number()
        print version_number
        q = raw_input("Would you like to change the version number for %s? y/n\n> " % self.apk_name)
        if q == "y":
            version_number = raw_input("What to?> ")
            self.set_version_number(version_number)

    def copy_symbols(self):
        artifacts_folder = os.path.join(self.proj_folder, 'bin', 'Release')
        stuff = os.listdir(artifacts_folder)
        msym_folder = None
        for name in stuff:
            if name.endswith(".mSYM"):
                msym_folder = os.path.join(artifacts_folder, name)
                break
        if msym_folder is not None:
            def zipdir(path, ziph):                
                for root, dirs, files in os.walk(path):
                    for file in files:
                        ziph.write(os.path.join(root, file),os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))
            msym_destination = os.path.join(os.path.expanduser("~/Desktop/"), os.path.basename(self.final_apk)) + ".mSYM.zip"
            zipf = zipfile.ZipFile(msym_destination, 'w', zipfile.ZIP_DEFLATED)
            zipdir(msym_folder, zipf)
            zipf.close()

    def run(self, update_versions=True, confirm_build=True):

        self.clean()

        self.final_apk = os.path.join(self.destination, "%s-" % self.apk_name)

        if update_versions:
            self.update_version()

        build_number = self.get_build_number()
        version_number = self.get_version_number()

        if confirm_build:
            print 'So thats version ' + version_number + " build " + build_number
            q = raw_input("Would you like to continue? y/n\n> ")
            if q != "y":
                print "Ok, not doing the build, suit yourself..."
                return None

        self.final_apk = self.final_apk + build_number + '-' + version_number + '.apk'

        print self.final_apk

        self.build()

        self.sign()

        self.copy_symbols()

        return self.final_apk
