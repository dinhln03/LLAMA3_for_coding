"""An example of jinja2 templating"""


from bareasgi import Application, HttpRequest, HttpResponse
import jinja2
import pkg_resources
import uvicorn

from bareasgi_jinja2 import Jinja2TemplateProvider, add_jinja2


async def http_request_handler(request: HttpRequest) -> HttpResponse:
    """Handle the request"""
    return await Jinja2TemplateProvider.apply(
        request,
        'example1.html',
        {'name': 'rob'}
    )


async def handle_no_template(request: HttpRequest) -> HttpResponse:
    """This is what happens if there is no template"""
    return await Jinja2TemplateProvider.apply(
        request,
        'notemplate.html',
        {'name': 'rob'}
    )

if __name__ == '__main__':

    TEMPLATES = pkg_resources.resource_filename(__name__, "templates")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES),
        autoescape=jinja2.select_autoescape(['html', 'xml']),
        enable_async=True
    )

    app = Application()
    add_jinja2(app, env)

    app.http_router.add({'GET'}, '/example1', http_request_handler)
    app.http_router.add({'GET'}, '/notemplate', handle_no_template)

    uvicorn.run(app, port=9010)
