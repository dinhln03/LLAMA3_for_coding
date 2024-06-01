#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains code to serve a web application to convert HTML to PDF.

This application uses a local install of the `wkhtmltopdf` binary for the conversion.
"""
import os
from subprocess import check_output
from tempfile import TemporaryDirectory

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route


async def execute_wkhtmltopdf(uri: str) -> bytes:
    """Run wkhtmltopdf on the command-line and return the output."""
    cmd = [
        "wkhtmltopdf",
        "--log-level",
        "none",
        uri,
        "-",
    ]
    return check_output(cmd)


async def convert_body(request: Request):
    """
    It's just _way_ easier to deal with files rather than STDIN.

    Take the body of the request, write it to a temporary file, then use
    wkhtmltopdf to convert it.
    """
    data = await request.body()

    if not data:
        return Response("ERROR: No body", status_code=400)

    with TemporaryDirectory() as tmpdirname:
        outfile = os.path.join(tmpdirname, "out.html")

        with open(outfile, "w") as fh:
            fh.write(data.decode("utf-8"))

        bytes = await execute_wkhtmltopdf(outfile)

    return Response(bytes, media_type="application/pdf")


async def convert_uri(request: Request):
    data = await request.json()

    if "uri" not in data:
        return Response("Invalid JSON in request", status_code=400)

    bytes = await execute_wkhtmltopdf(data["uri"])
    return Response(bytes, media_type="application/pdf")


app = Starlette(
    debug=True,
    routes=[
        Route("/uri", convert_uri, methods=["POST"]),
        Route("/data", convert_body, methods=["POST"]),
    ],
)
