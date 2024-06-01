import unittest
from typing import Optional
from pathlib import Path
from time import strftime, gmtime
from github import Github
import json
import yaml
from .. import get_repo_meta, clone_and_archive
from ..get_github import dump_list
from pathlib import Path

class TestGetGithub(unittest.TestCase):
    """
        Test coverage for the get_github module
    """
    _current_dir = Path(__file__).resolve().parent

    #BORROWED from https://github.com/NOAA-OWP/DMOD/blob/master/python/lib/scheduler/dmod/test/it_redisManager.py
    @classmethod
    def find_project_root_directory(cls, current_directory: Optional[Path]) -> Optional[Path]:
        """
        Given a directory (with ``None`` implying the current directory) assumed to be at or under this project's root,
        find the project root directory.
        This implementation attempts to find a directory having both a ``.git/`` child directory and a ``.env`` file.
        Parameters
        ----------
        current_directory
        Returns
        -------
        Optional[Path]
            The project root directory, or ``None`` if it fails to find it.
        """
        if not current_directory:
            current_directory = TestGetGithub._current_dir
        abs_root = Path(current_directory.absolute().root)
        while current_directory.absolute() != abs_root:
            if not current_directory.is_dir():
                current_directory = current_directory.parent
                continue
            git_sub_dir = current_directory.joinpath('.git')
            child_env_file = current_directory.joinpath('config.yaml')
            if git_sub_dir.exists() and git_sub_dir.is_dir() and child_env_file.exists() and child_env_file.is_file():
                return current_directory
            current_directory = current_directory.parent
        return None

    @classmethod
    def load_token(cls):
        """
            Read an API token from a configuration file, if none found, use '' for no auth
        """
        token = ''
        root_dir = cls.find_project_root_directory(None)
        if not root_dir:
            return token

        config_file = root_dir/'config.yaml'
        if config_file.exists():
            with open(config_file) as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
                try:
                    token = config['token']
                except:
                    print("Unable to load api-token from project root directory config.yaml")

        return token

    def setUp(self):
        self.token = self.load_token()
        if self.token:
            #Token auth github, higher rate limit
            self.github = Github(self.token)
        else:
            #Anonimous github, severly rate limited API
            self.github = Github()
        self.org_string = 'NOAA-OWP'
        self.repo_string = 'owp-open-source-project-template'
        self.repo_w_wiki = 'DMOD'
        self.org = self.github.get_organization(self.org_string)
        self.wiki_repo = self.org.get_repo(self.repo_w_wiki)
        self.repo = self.org.get_repo(self.repo_string)
        self.time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())

    def tearDown(self):
        for p in Path(TestGetGithub._current_dir).glob("*.json"):
            if p.is_file():
                p.unlink()
        for p in Path(TestGetGithub._current_dir).glob("*.tar.gz"):
            if p.is_file():
                p.unlink()

    def test_get_repo_meta(self):
        """
            Test the archive_repo function to ensure all meta data is properly captured
        """
        meta = get_repo_meta(self.repo, self.time, TestGetGithub._current_dir)
        self.assertIsNotNone(meta)
        self.assertTrue(len(meta), 6)
        #defer name substitution
        pattern =  "{repo}_{name}_{time}.json".format(repo=self.repo_string, name="{name}", time=self.time)
        self.assertEqual(meta[0].name, pattern.format(name='comments'))
        self.assertEqual(meta[1].name, pattern.format(name='issues'))
        self.assertEqual(meta[2].name, pattern.format(name='issue_comments'))
        self.assertEqual(meta[3].name, pattern.format(name='pulls'))
        self.assertEqual(meta[4].name, pattern.format(name='pulls_comments'))
        self.assertEqual(meta[5].name, pattern.format(name='pulls_review_comments'))

        self.assertTrue((TestGetGithub._current_dir/pattern.format(name='comments')).exists())
        self.assertTrue((TestGetGithub._current_dir/pattern.format(name='issues')).exists())
        self.assertTrue((TestGetGithub._current_dir/pattern.format(name='issue_comments')).exists())
        self.assertTrue((TestGetGithub._current_dir/pattern.format(name='pulls')).exists())
        self.assertTrue((TestGetGithub._current_dir/pattern.format(name='pulls_comments')).exists())
        self.assertTrue((TestGetGithub._current_dir/pattern.format(name='pulls_review_comments')).exists())

    def test_clone_and_archive(self):
        """
            Test the clone functionality
        """
        #Sorta hackily testing the archive_repo logic here...FIXME later
        #FIXME has_wiki is broke!!!
        self.assertFalse(self.repo.has_wiki)
        clone_url = self.repo.clone_url
        archive_name = clone_and_archive(self.repo_string, clone_url, self.time, TestGetGithub._current_dir, [])
        name = '{repo}_github_archive_{time}.tar.gz'.format(repo=self.repo_string, time=self.time)
        self.assertEqual(archive_name.name, name)
        self.assertTrue((TestGetGithub._current_dir/name).exists())
        #TODO test existence of repo in archive

    def test_clone_and_archive_1(self):
        """
            Test cloning a repo with a wiki
        """
        #Sorta hackily testing the archive_repo logic here...FIXME later
        self.assertTrue(self.wiki_repo.has_wiki)
        wiki_url = self.wiki_repo.clone_url[:-3]+'wiki.git'

        #finally get the repo code itself
        clone_url = self.wiki_repo.clone_url
        archive_name = clone_and_archive(self.repo_w_wiki, clone_url, self.time, TestGetGithub._current_dir, [], wiki_url)

        name = '{repo}_github_archive_{time}.tar.gz'.format(repo=self.repo_w_wiki, time=self.time)
        self.assertEqual(archive_name.name, name)
        self.assertTrue((TestGetGithub._current_dir/name).exists())
        #TODO test existense of wiki in archive

    @unittest.skip("Incomplete mock implementation for dumped item")
    def test_dump_list(self):
        pass
        #Quick json format test for list objects
        outfile = dump_list("test_repo", self.time, TestGetGithub._current_dir, "test_key", [ '{json=dict, test=values}', 'json=dict2, test=values2}'])
        with open(outfile) as fp:
            data = json.load(fp)
            self.assertTrue(data['json'], 'dict')
