#!/usr/bin/env python
# Author: Nick Zwart
# Date: 2016jun01
#   Backup all the projects of a git-hub style website via git mirroring.
#   https://www.garron.me/en/bits/backup-git-bare-repo.html 

import os
import sys
import time
import gitlab # external GitLab API
import github # external GitHub API
import shutil
import hashlib
import optparse
import subprocess

class GitWebsiteTypeAPI:
    '''The abstract class to template each git-based website api.
    '''
    def __init__(self, token, url):
        self._token = token
        self._url = url

    def numProjects(self):
        # return the number of projects
        pass

    def projectPath(self, index):
        # return the full path for each project including group i.e.
        #   <user/group-directory>/<repository-name>
        # e.g.
        #   nckz/BackupHub
        pass

    def projectURL(self, index):
        # return the ssh-url that assumes ssh-keys have been distributed e.g.
        #   git@git<lab/hub>.com:<user/group>/<repo-name>.git
        # e.g.
        #   git@github.com:nckz/BackupHub.git
        pass

class GitLabAPI(GitWebsiteTypeAPI):
    def __init__(self, token, url):
        GitWebsiteTypeAPI.__init__(self, token, url)

        # authenticate a gitlab session
        self._gl = gitlab.Gitlab(self._url, self._token)
        self._gl.auth()

        # list all projects
        self._projects = self._gl.projects.list(all=True)

    def numProjects(self):
        return len(self._projects)

    def projectPath(self, index):
        return self._projects[index].path_with_namespace

    def projectURL(self, index):
        return self._projects[index].ssh_url_to_repo

class GitHubAPI(GitWebsiteTypeAPI):
    def __init__(self, token, url=''):
        GitWebsiteTypeAPI.__init__(self, token, url)

        # authenticate a gitlab session
        self._gh = github.Github(self._token)

        # list all projects
        self._projects = self._gh.get_user().get_repos()

    def numProjects(self):
        return len([i for i in self._projects])

    def projectPath(self, index):
        return self._projects[index].full_name

    def projectURL(self, index):
        return self._projects[index].ssh_url

class GitBareMirror:
    '''A simple git interface for managing bare-mirroed repos that backup url
    accessible upstream repos.
    '''

    def __init__(self, path, url, overwrite=False, moveAside=False):
        self._path = path
        self._origin_url = url
        self._overwrite = overwrite
        self._moveAside = moveAside

        if self.validExistingRepo():
            self.update()
        else:
            self.createMirroredRepo()

    def validExistingRepo(self):

        try:
            assert os.path.isdir(self._path), ('The supplied directory '
                    'does not exist.')

            # move to the existing repo and check if its bare
            os.chdir(self._path)
            cmd = subprocess.Popen('git rev-parse --is-bare-repository',
                    shell=True, stdout=subprocess.PIPE)
            cmd.wait()

            # Error checking
            assert cmd.returncode != 128, ('The supplied directory '
                    'exists, but is not a git repo.')
            assert cmd.returncode == 0, 'There was an unhandled git error.'
            firstline = cmd.stdout.readlines()[0].decode('utf8')
            assert 'false' not in firstline, ('The supplied directory '
                    'is NOT a bare repo.')
            assert 'true' in firstline, ('Unable to verify that the repo is '
                    'bare.')

            # check if the existing repo has the same origin url
            # -prevent name collision if group/org namespace isn't used
            cmd = subprocess.Popen('git config --get remote.origin.url',
                    shell=True, stdout=subprocess.PIPE)
            cmd.wait()
            firstline = cmd.stdout.readlines()[0].decode('utf8')

            assert self._origin_url in firstline, ('The existing repo '
                    'has a url that differs from the supplied origin url.')

            return True

        except AssertionError as err:
            print('The given path does not contain a valid repo by:', err)
            return False

    def update(self):
        cmd = subprocess.Popen('git remote update', shell=True,
                stdout=subprocess.PIPE)
        cmd.wait()

        assert cmd.returncode == 0, 'ERROR: git error'
        print('SUCCESS (updated)')

    def createMirroredRepo(self):

        # Handle existing directories based on user options:
        # move the dir to a unique name, remove it, or fail w/ exception
        if self._moveAside and os.path.exists(self._path):
            parentPath = os.path.dirname(self._path)
            dirContents = str(os.listdir(parentPath)).encode('utf8')
            newNameExt = hashlib.md5(dirContents).hexdigest()
            newName = self._path+'_'+newNameExt+'_bu'
            print('MOVING PATH', self._path, newName)
            shutil.move(self._path, newName)
        elif self._overwrite and os.path.exists(self._path):
            print('REMOVING PATH', self._path)
            shutil.rmtree(self._path)
        else:
            assert not os.path.exists(self._path), ('ERROR: the supplied path '
                    'already exists, unable to create mirror.') 

        os.makedirs(self._path)
        os.chdir(self._path)
        cmd = subprocess.Popen('git clone --mirror ' + str(self._origin_url)
                + ' .', shell=True, stdout=subprocess.PIPE)
        cmd.wait()
        print('SUCCESS (new mirror)')

if __name__ == '__main__':

    # parse input args
    parser = optparse.OptionParser()
    parser.add_option('--path', dest='backupPath', action='store',
            type='string', default=os.path.expanduser('~/backup'),
            help='The directory to store the backups.')
    parser.add_option('--ignore-errors', dest='ignoreErrors',
            action='store_true', default=False,
            help='Continue to backup other repos if one has failed.')
    parser.add_option('--overwrite', dest='overwrite',
            action='store_true', default=False,
            help='Overwrite existing directories.')
    parser.add_option('--move-aside', dest='moveAside',
            action='store_true', default=False,
            help='Move existing directories aside with a tempfile extension.')
    parser.add_option('--token', dest='token', action='store',
            type='string', default=None,
            help='The token required to access the target git web api.')
    parser.add_option('--website', dest='website', action='store',
            type='string', default=None,
            help='The hub website where the git repos are stored.')
    parser.add_option('--github', dest='github',
            action='store_true', default=False,
            help='Connect to GitHub.')
    parser.add_option('--gitlab', dest='gitlab',
            action='store_true', default=True,
            help='Connect to GitLab (default).')

    options, args = parser.parse_args(sys.argv)

    localtime = time.asctime( time.localtime(time.time()) )
    print("BackupHub Start:", localtime)

    assert options.token is not None

    if options.github:
        options.gitlab = False

    if options.gitlab:
        assert options.website is not None

    # Check for existing backup directory and make one if it doesn't exist.
    if not os.path.isdir(options.backupPath):
        print('The specified backup path doesn\'t exist.')
        sys.exit(1)

    # Get the repository info from the git web api.
    if options.github:
        webapi = GitHubAPI(options.token)
    elif options.gitlab:
        webapi = GitLabAPI(options.token, options.website)

    # Display whats going on as the repos get either updated or newly mirrored.
    print('Repository:')
    for i in range(webapi.numProjects()):
        try:
            curPath = os.path.join(options.backupPath, webapi.projectPath(i))
            curURL = webapi.projectURL(i)
            print('\nSyncing: ', curURL, curPath)
            repo = GitBareMirror(curPath, curURL, overwrite=options.overwrite,
                    moveAside=options.moveAside)
        except Exception as err:
            if options.ignoreErrors:
                print(err)
            else:
                raise

    localtime = time.asctime( time.localtime(time.time()) )
    print("BackupHub Finished:", localtime)
