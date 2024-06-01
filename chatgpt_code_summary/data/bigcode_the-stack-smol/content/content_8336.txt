import os

from indexing.pathanalyzer import PathAnalyzer
from indexing.pathanalyzerstore import PathAnalyzerStore

class Indexer:
    """
    Traverses the given directory using the DFS algorithm. Allows registering different rules for handling different
    file types and calls the associated PathAnalyzers and Collectors indirectly for each type.
    """

    ####################################################################################################################
    # Constructor.
    ####################################################################################################################

    def __init__(self, max_depth=10):
        """
        Initializes attributes and checks the maximum depth provided.

        Parameters
        ----------
        max_depth : int
            The maximum depth to look in.
        """

        ### Validate parameters.
        if max_depth < 1:
            raise Exception('max_depth must be greater than or equal to 1.')

        ### Attributes from outside.
        self._max_depth = max_depth

        ### Private attributes.
        # A collection of analyzers which handle different file types.
        self._analyzers = []
        # The depth we are currently in.
        self._current_depth = 0
        # The list of directories to index.
        self._rules = {}

    ####################################################################################################################
    # Public methods.
    ####################################################################################################################

    def add_rule(self, directory, policy):
        """
        Registers a new directory to index. Does nothing if the given directory is already added.

        Parameters
        ----------
        directory : str
            The directory to be indexed.
        policy : IndexerPolicy
            A policy that applies to this directory.
        """

        analyzer = self._create_analyzer(policy)
        analyzer_store = self._create_analyzerstore(directory)

        analyzer_store.add_analyzer(policy.extensions, analyzer)

    def index(self):
        """
        Initializes filters, initiates indexing and after the indexing process has finished, cleans filters.
        """

        for analyzer in self._analyzers:
            analyzer.init_filters()

        for directory, analyzer_store in self._rules.items():
            if os.path.exists(directory):
                self._scan_directory(directory, analyzer_store)

        for analyzer in self._analyzers:
            analyzer.clean_filters()

    ####################################################################################################################
    # Auxiliary methods.
    ####################################################################################################################

    def _analyze_file(self, current_path, analyzer_store):

        current_path_without_extension, current_extension = os.path.splitext(current_path)

        analyzer = analyzer_store.find_analyzer(current_extension)
        if analyzer is not None:
            analyzer.analyze(current_path_without_extension, current_extension)

    def _create_analyzer(self, policy):

        analyzer = PathAnalyzer(policy)
        self._analyzers.append(analyzer)

        return analyzer

    def _create_analyzerstore(self, directory):

        if directory not in self._rules:
            self._rules[directory] = PathAnalyzerStore()

        return self._rules[directory]

    def _enter(self, directory):
        """
        Indicates for the analyzers that we entered into the given directory.

        Parameters
        ----------
        directory : str
            The directory we entered.
        """

        for analyzer in self._analyzers:
            analyzer.enter(directory)

        self._current_depth = self._current_depth + 1

    def _leave(self):
        """
        Indicates for the analyzers that we are leaving the last directory.
        """

        for analyzer in self._analyzers:
            analyzer.leave()

        self._current_depth = self._current_depth - 1


    def _scan_directory(self, path, analyzer_store):
        """
        Does the real indexing. Iterates through the directory using DFS, and invokes the registered analyzers to
        analyze and store the data.

        Parameters
        ----------
        path : str
            The path to enumerate.
        analyzers : PathAnalyzerStore
            The PathAnalyzerStore to use.
        """

        for current_file in os.listdir(path):

            current_path = os.path.join(path, current_file)

            if self._current_depth >= self._max_depth:
                return

            if os.path.isdir(current_path):
                self._enter(current_file)
                self._scan_directory(current_path, analyzer_store)
                self._leave()
            else:
                self._analyze_file(current_path, analyzer_store)
