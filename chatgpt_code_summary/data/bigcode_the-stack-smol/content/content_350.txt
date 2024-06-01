import urllib2
from zope.interface import implements

from plone.portlets.interfaces import IPortletDataProvider
from plone.app.portlets.portlets import base
from Products.CMFCore.utils import getToolByName

from zope import schema
from zope.formlib import form

from Products.Five.browser.pagetemplatefile import ViewPageTemplateFile

from wad.blog.utils import find_portlet_assignment_context
from wad.blog.blogentry import IBlogEntry
from wad.blog import MessageFactory as _


class IBlogCategoriesPortlet(IPortletDataProvider):
    """A portlet

    It inherits from IPortletDataProvider because for this portlet, the
    data that is being rendered and the portlet assignment itself are the
    same.
    """
    archive_view = schema.TextLine(
        title=_(u"Archive view"),
        description=_(u"The name of the archive view"),
        default=u'blog-view',
        required=True
    )


class Assignment(base.Assignment):
    """Portlet assignment.

    This is what is actually managed through the portlets UI and associated
    with columns.
    """

    implements(IBlogCategoriesPortlet)

    def __init__(self, archive_view=u'blog-view'):
        self.archive_view = archive_view

    @property
    def title(self):
        """This property is used to give the title of the portlet in the
        "manage portlets" screen.
        """
        return _("Categories")


class Renderer(base.Renderer):
    """Portlet renderer.

    This is registered in configure.zcml. The referenced page template is
    rendered, and the implicit variable 'view' will refer to an instance
    of this class. Other methods can be added and referenced in the template.
    """

    render = ViewPageTemplateFile('categories.pt')

    def keywords(self):
        catalog = getToolByName(self.context, 'portal_catalog')
        keywords = catalog.uniqueValuesFor('Subject')
        keywords = [unicode(k, 'utf-8') for k in keywords]
        return keywords

    def archive_url(self, subject):
        # Get the path of where the portlet is created. That's the blog.
        assignment_context = find_portlet_assignment_context(self.data,
                                                             self.context)
        if assignment_context is None:
            assignment_context = self.context
        self.folder_url = assignment_context.absolute_url()
        sub = urllib2.quote(subject.encode('utf-8'))
        url = '%s/%s?category=%s' % (self.folder_url,
                                     self.data.archive_view,
                                     sub)
        return url

    def blog_url(self):
        assignment_context = find_portlet_assignment_context(self.data,
                                                             self.context)
        if assignment_context is None:
            assignment_context = self.context
        return assignment_context.absolute_url()

    def count_entries(self, subject):
        catalog = getToolByName(self.context, 'portal_catalog')
        brains = catalog(object_provides=IBlogEntry.__identifier__,
                         Subject=subject.encode('utf-8'))
        return len(brains)

    def count_all_entries(self):
        catalog = getToolByName(self.context, 'portal_catalog')
        brains = catalog(object_provides=IBlogEntry.__identifier__)
        return len(brains)


class AddForm(base.AddForm):
    """Portlet add form.

    This is registered in configure.zcml. The form_fields variable tells
    zope.formlib which fields to display. The create() method actually
    constructs the assignment that is being added.
    """
    form_fields = form.Fields(IBlogCategoriesPortlet)

    def create(self, data):
        return Assignment(**data)


class EditForm(base.EditForm):
    """Portlet edit form.

    This is registered with configure.zcml. The form_fields variable tells
    zope.formlib which fields to display.
    """
    form_fields = form.Fields(IBlogCategoriesPortlet)
