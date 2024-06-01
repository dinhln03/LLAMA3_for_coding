import dash_bootstrap_components as dbc
import dash_html_components as html
from utils import UrlIndex


def NavBar(pathname=None):
    return dbc.Nav([
        dbc.NavItem(
            dbc.NavLink(
                html.H4("Home", className='mt-1', style={'color': (
                    'white' if pathname == UrlIndex.HOME.value or pathname == UrlIndex.ROOT.value else 'black')}),
                active=(pathname == UrlIndex.HOME.value or pathname == UrlIndex.ROOT.value),
                href=UrlIndex.HOME.value
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                html.H4("Plot", className='mt-1',
                        style={'color': ('white' if pathname == UrlIndex.PLOT.value else 'black')}),
                active=(pathname == UrlIndex.PLOT.value),
                href=UrlIndex.PLOT.value
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                html.H4("Help", className='mt-1',
                        style={'color': ('white' if pathname == UrlIndex.HELP.value else 'black')}),
                active=(pathname == UrlIndex.HELP.value),
                href=UrlIndex.HELP.value
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                html.H4("Get in touch", className='mt-1',
                        style={'color': ('white' if pathname == UrlIndex.CONTACT.value else 'black')}),
                active=(pathname == UrlIndex.CONTACT.value),
                href=UrlIndex.CONTACT.value
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                html.H4("Rigden Lab", className='mt-1',
                        style={'color': ('white' if pathname == UrlIndex.RIGDEN.value else 'black')}),
                active=(pathname == UrlIndex.RIGDEN.value),
                href=UrlIndex.RIGDEN.value
            )
        )
    ], pills=True, fill=True, justified=True, style={'border-bottom': '2px solid', 'border-top': '2px solid'}
    )


def Footer(fixed=False):
    if fixed:
        style = {'position': 'fixed', 'background': 'gray', 'width': '100%', 'bottom': '0px'}
    else:
        style = {}

    return html.Div([
        html.Br(),
        html.Br(),
        html.Br(),
        dbc.Navbar([
            dbc.Row(
                dbc.Col([
                    html.Small('Hosted by', className='mr-3'),
                    html.A(html.Img(src='/assets/ccp4-online_logo.png', height='35hv'), href=UrlIndex.CCP4_ONLINE.value)
                ], width=12), className='text-center container-fluid'
            )
        ], className='footer py-3', style=style)
    ])
