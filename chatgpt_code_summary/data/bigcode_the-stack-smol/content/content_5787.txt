from urllib.parse import quote
import re


def parse_equation(match):
    # Converts a latex expression into something the tex API can understand
    eq = match.group(0)

    # Curly brackets need to be escaped
    eq = eq.replace('{', '\{')
    eq = eq.replace('}', '\}')

    # Create the url using the quote method which converts special characters
    url = 'https://tex.s2cms.ru/svg/%s' % quote(eq)

    # Return the markdown SVG tag
    return '![](%s)' % url


def parse_markdown(md):
    # Define a pattern for catching latex equations delimited by dollar signs
    eq_pattern = r'(\$.+?\$)'

    # Substitute any latex equations found
    return re.sub(eq_pattern, parse_equation, md)


def markdown_texify(file_in, file_out):
    # Read input file
    markdown = open(file_in).read()

    # Parse markdown, take care of equations
    latex = parse_markdown(markdown)

    # Write to out-file
    result = open(file_out, 'w').write(latex)
    print('Finished, %i characters written to %s' % (result, file_out))

