

from flask_appbuilder import BaseView, expose
from config import APP_ICON, APP_NAME
   
from flask import g
 
def get_user():
    return g.user

def custom_template():
    app_name = "GEA"
    app_version = "1.2"

    return app_name, app_version


class someView(BaseView):
    """
        A simple view that implements the index for the site
    """    
    route_base = ''
    default_view = 'index'
    index_template = 'appbuilder/index.html'
   
    @expose('/')
    def index(self):
         
        from app import db
        from .models import Partner, Unit, Application, Doctype
        
        session = db.session
        partner = session.query(Partner).count()
        unit = session.query(Unit).count()
        application = session.query(Application).count()
        doctype = session.query(Doctype).count()
        
        self.update_redirect()
        return self.render_template(self.index_template,
                                    appbuilder=self.appbuilder,
                                    partner=partner,
                                    unit=unit,
                                    material=application,
                                    doctype=doctype,
                                    user=g.user)


class MyIndexView(someView):
    index_template = 'index.html'

