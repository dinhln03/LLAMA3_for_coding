import re

def camel_space(string):
    string = re.sub(r'(?<!^)(?=[A-Z])', ' ', string)
    return string

Test.assert_equals(solution("helloWorld"), "hello World")
Test.assert_equals(solution("camelCase"), "camel Case")
Test.assert_equals(solution("breakCamelCase"), "break Camel Case")