from dyc.utils import (
    get_leading_whitespace,
    read_yaml,
    get_indent_forward,
    get_indent_backward,
    get_extension,
    is_comment,
)


class TestGetLeadingWhitespace:
    def test_tabs(self):
        """Test tabs functionality"""
        text = '\t\tHello'
        expected = '\t\t'
        got = get_leading_whitespace(text)
        assert expected == got

    def test_whitespace(self):
        """Test whitespace functionality"""
        space = '                '
        text = '{space}Such a long whitespace'.format(space=space)
        expected = space
        got = get_leading_whitespace(text)
        assert expected == got


class TestReadYaml:
    def test_should_return_none_if_not_found(self):
        random_path = '/path/to/non/existing/file.yaml'
        expected = None
        got = read_yaml(random_path)
        assert expected == got


class TestGetIndentForward:
    def test_forward(self):
        lines = []
        lines.append( '\n')
        lines.append('This is a Test')
        assert get_indent_forward(lines, 0) == '\n'

class TestGetIndentBackward:
    def test_backward(self):
        lines = []
        lines.append( '\n')
        lines.append('This is a Test')
        assert get_indent_backward(lines, 1) == 'This is a Test'
        
class TestGetExtension:
    def test_existing_extension_valid(self):
        ext = 'file.puk'
        expected = 'puk'
        got = get_extension(ext)
        assert expected == got

    def test_non_existing_extension(self):
        ext = 'file'
        expected = ''
        got = get_extension(ext)
        assert expected == got

    def test_wrong_extension_type(self):
        exts = [dict(), False, True, [], 123]
        expected = ''
        for ext in exts:
            got = get_extension(ext)
            assert expected == got


class TestIsComment:
    def test_valid_comments(self):
        """Testing valid comments"""
        text = '# Hello World'
        assert is_comment(text, ['#']) == True

    def test_invalid_comments(self):
        """Testing invalid comments"""
        text = '# Hello World'
        assert is_comment(text, ['//']) == False

class UtilsTest():
    def __init__(self, whitespace, read_yaml, extension, comment,
                 indent_forward, indent_backward):
      self.test_get_leading_white_space = whitespace
      self.test_read_yaml =  read_yaml
      self.test_get_extension = extension
      self.test_is_comment = comment
      self.test_get_indent_forward = indent_forward
      self.test_get_indent_backward = indent_backward

    def test_whitespace(self):
      self.test_get_leading_white_space.test_tabs()
      self.test_get_leading_white_space.test_whitespace()

    def test_readYaml(self):
        self.test_read_yaml.test_should_return_none_if_not_found()

    def test_extension(self):
        self.test_get_extension.test_existing_extension_valid()
        self.test_get_extension.test_non_existing_extension()
        self.test_get_extension.test_wrong_extension_type()

    def test_comment(self):
        self.test_is_comment.test_valid_comments()
        self.test_is_comment.test_invalid_comments()

    def test_indent_forward(self):
        self.test_get_indent_forward.test_forward()

    def test_indent_backward(self):
        self.test_get_indent_backward.test_backward()


utils_test = UtilsTest(TestGetLeadingWhitespace(),
                       TestReadYaml(),
                       TestGetExtension(),
                       TestIsComment(),
                       TestGetIndentForward(),
                        TestGetIndentBackward())

utils_test.test_whitespace()
utils_test.test_readYaml()
utils_test.test_extension()
utils_test.test_comment()
utils_test.test_indent_forward()
utils_test.test_indent_backward()
    
