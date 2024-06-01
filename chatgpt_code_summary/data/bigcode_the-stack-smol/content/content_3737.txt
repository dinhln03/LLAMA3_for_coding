#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import builderutils.parser as parser
import builderutils.renderer as renderer
import builderutils.dom as dom


@click.group()
def cli():
    pass


@click.command()
@click.option("--configfile", type=click.Path(), help="Builder config", required=True)
def create(configfile):
    print("create command!")
    parserObj = parser.ConfigParser(configfile)
    print("Parser Obj: ", parserObj)
    domObj = dom.DomManager(parserObj)
    domObj.buildDomTree()
    dom.DomManager.parseDomTree(dom.SAMPLE_DOM)

    # parserObj = parser.BuilderParser(configfile)
    # renderObj = renderer.Renderer()
    # renderObj.build_staging_environment(parserObj.parsedData)

    # userConfig = parserObj.parsedData["user_config"]
    # htmlTemplate = parserObj.parsedData["html_template"]
    # flaskTemplate = parserObj.parsedData["flask_template"]

    # renderObj.build_html_documents(htmlTemplate, userConfig)
    # renderObj.build_flask_app(flaskTemplate, userConfig)


def main():
    cli.add_command(create)
    cli()


if __name__ == "__main__":
    main()
