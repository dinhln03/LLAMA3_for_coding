"""
The MIT License (MIT)

Copyright (c) 2017 Andreas Poppele
Copyright (c) 2017 Roland Jaeger

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


from ..scrabTask import FileTask

import os

name = "LanguageDetector"
version = "1.1.1"


class LanguageDetector(FileTask):

    cpp_extensions = ['.cpp', '.c++', '.cc',
                      '.cxx', '.c', '.h', '.hpp', '.hxx']
    c_extensions = ['.c', '.h']
    rust_extensions = ['.rs']
    ruby_extensions = ['.rb']
    java_extensions = ['.java']
    go_extensions = ['.go']
    php_extensions = ['.php', '.phtml', '.php3', '.php4', '.php5', '.php7',
                      '.phps']
    js_extensions = ['.js']
    objective_c_extensions = ['.h', '.m', '.mm', '.C']
    swift_extensions = ['.swift']
    c_sharp_extensions = ['.cs']
    python_extensions = ['.py']

    """
    Tries to detect the programming language of a library based on the file
    extension

    Example:
        LanguageDetector:
          main_language: C
          languages:
          - C
          - C++
          - Python

    :param    task_params:  Parameter given explicitly for this task, for all
                            projects, defined in the task.yaml
    :param    global_args:  Arguments that will be passed to all tasks. They
                            _might_ contain something that is useful for the
                            task, but the task has to check if it is _there_ as
                            these are user provided. If they are needed to work
                            that check should happen in the argHandler.
    """

    def __init__(self, parameter, global_args):
        super(LanguageDetector, self).__init__(name, version, parameter,
                                               global_args)
        # dictionary containing the common file extensions
        # for each of the languages
        self.__language_extensions = self.__get_language_extensions()
        self.__report = self.__get_files_per_language()

    def __get_language_extensions(self):
        """
        :returns: A directory of the considered language extensions
        """
        return {
            'C++':
                self.cpp_extensions,
            'C':
                self.c_extensions,
            'Rust':
                self.rust_extensions,
            'Ruby':
                self.ruby_extensions,
            'Java':
                self.java_extensions,
            'Go':
                self.go_extensions,
            'PHP':
                self.php_extensions,
            'JavaScript':
                self.js_extensions,
            'Objective-C':
                self.objective_c_extensions,
            'Swift':
                self.swift_extensions,
            'C#':
                self.c_sharp_extensions,
            'Python':
                self.python_extensions
        }

    def __get_files_per_language(self):
        """
        :returns: A default directory of the considered languages, their
                  extensions and the amount of files that have that extension
                  (default=0)
        """
        return {
            'C++':
                {extension: 0 for extension in self.cpp_extensions},
            'C':
                {extension: 0 for extension in self.c_extensions},
            'Rust':
                {extension: 0 for extension in self.rust_extensions},
            'Ruby':
                {extension: 0 for extension in self.ruby_extensions},
            'Java':
                {extension: 0 for extension in self.java_extensions},
            'Go':
                {extension: 0 for extension in self.go_extensions},
            'PHP':
                {extension: 0 for extension in self.php_extensions},
            'JavaScript':
                {extension: 0 for extension in self.js_extensions},
            'Objective-C':
                {extension: 0 for extension in self.objective_c_extensions},
            'Swift':
                {extension: 0 for extension in self.swift_extensions},
            'C#':
                {extension: 0 for extension in self.c_sharp_extensions},
            'Python':
                {extension: 0 for extension in self.python_extensions},
        }

    def __decide_h_extension(self):
        """
        Decides which language 'owns' how many .h files

        :returns: The report with divided header files
        """
        report = self.__report
        h_files = report['C']['.h']
        if h_files > 0:
            c_files = (sum(report['C'].values()) - h_files)
            cpp_files = (sum(report['C++'].values())
                         - h_files
                         - report['C++']['.c'])
            oc_files = (
                sum(report['Objective-C'].values()) - h_files)
            lang_fiels = c_files + cpp_files + oc_files

            # Header only libraries are 'common' in C and C++
            # the benefit of doubt goes to C
            if lang_fiels == 0:
                report['C']['.h'] = 1
                report['C++']['.h'] = 0
                report['Objective-C']['.h'] = 0
            else:
                report['C']['.h'] = (h_files *
                                     c_files / lang_fiels)
                report['C++']['.h'] = (h_files *
                                       cpp_files / lang_fiels)
                report['Objective-C']['.h'] = (h_files *
                                               oc_files / lang_fiels)
        return report

    def __calculate_main_language(self, report):
        """
        Calculates the main language (maximum of files extensions)

        :param    report:  The report

        :returns: The main language.
        """
        max_files = 0
        max_lang = None

        for language in report:
            lang_fiels = sum(report[language].values())
            if max_files < lang_fiels:
                max_lang = language
                max_files = lang_fiels

        return max_lang

    def __calculate_used_languages(self, report):
        """
        Calculates the used languages by throwing away the extension counts and
        collapsing them to the language. Only languages that have at least one
        file extension are kept and will appear in the report

        :param    report:  The report

        :returns: The used languages.
        """
        languages = {}

        for language in report:
            total_files = sum(report[language].values())

            if total_files > 0:
                languages[language] = total_files

        return sorted(languages, key=languages.get, reverse=True)

    def scrab(self, project, filepath, file):
        """
        Counts the files that have an extension of one of the languages

        :param    project:   The project that the scrab task shall analyse
        :param    filepath:  The filepath to the file that can be analysed
        :param    file:      The file as string that can be analysed

        :returns: Report that contains the scrabbed information of *this* file
                  - the extensions have either a count of 0 or 1
        """
        filename, file_extension = os.path.splitext(filepath)

        for language in self.__language_extensions:
            if file_extension in self.__language_extensions[language]:
                self.__report[language][file_extension] += 1

    def report(self):
        """
        Decides which headers files are (probable) from which language,
        calculates the main language and  removes redundant / unnecessary
        detailed information from the report

        :param    report:  The complete report this task created

        :returns: Report that contains all scrabbed information
                  eg.:
                  LanguageDetector:
                      main_language: C
                      languages:
                      - C
                      - C++
                      - Python
        """
        pre_report = self.__decide_h_extension()
        main_language = self.__calculate_main_language(pre_report)

        # write the result to the report
        report = {}
        report['main_language'] = main_language
        report['languages'] = self.__calculate_used_languages(pre_report)

        return report
