from django.contrib import admin
from django.contrib.auth.models import User
from django.test.testcases import TestCase
from django.urls import reverse

from pagetools.menus.admin import MenuAdmin, make_entrieable_admin
from pagetools.menus.apps import MenusConfig
from pagetools.menus.models import Link, Menu, MenuEntry
from pagetools.tests.test_models import ConcretePublishableLangModel
from pagetools.utils import get_adminedit_url
from pagetools.widgets.settings import TEMPLATETAG_WIDGETS


class CPMAdmin(admin.ModelAdmin):
    model = ConcretePublishableLangModel


admin.site.register(ConcretePublishableLangModel, CPMAdmin)


class MenuAdminTests(TestCase):
    def setUp(self):
        self.admin = User.objects.create_superuser("admin", "q@w.de", "password")
        self.client.login(username="admin", password="password")
        self.site = admin.site

    def _data_from_menu(self, menu):
        return {
            key: menu.__dict__[key]
            for key in (
                "id",
                "lang",
                "title",
                "slug",
                "content_type_id",
                "object_id",
                "enabled",
                "lft",
                "rght",
                "tree_id",
                "level",
            )
        }

    def test_admin_index(self):
        """ test index because customdashboard with MenuModule is may used"""
        adminindex = reverse("admin:index")
        response = self.client.get(adminindex, follow=True, extra={"app_label": "admin"})
        self.assertIn(response.status_code, (200, 302))

    def test_add(self):
        adminurl = reverse("admin:menus_menu_add", args=[])
        self.client.post(adminurl, {"title": "Menu1"})
        menu = Menu.objects.get(title="Menu1")
        self.assertEqual(len(menu.children.all()), 0)
        return menu

    def test_update(self):
        menu = Menu.objects.add_root(title="Menu1")
        entries = []
        for i in range(1, 3):
            entries.append(
                MenuEntry.objects.add_child(
                    parent=menu,
                    title="e%s" % i,
                    content_object=Link.objects.create(
                        url="#%s" % i,
                    ),
                    enabled=True,
                )
            )
        adminurl = reverse("admin:menus_menu_change", args=[menu.pk])
        self.client.get(adminurl, {"pk": menu.pk})
        data = self._data_from_menu(menu)
        data["entry-order-id-0"] = entries[0].pk
        data["entry-text-0"] = "changed"
        data["entry-published-0"] = 1
        self.client.post(adminurl, data)
        children = menu.children_list()
        self.assertEqual(children[0]["entry_title"], "changed")

    def test_reorder(self):
        menu = Menu.objects.add_root(title="Menu1")
        entries = []
        for i in range(1, 3):
            entries.append(
                MenuEntry.objects.add_child(
                    parent=menu,
                    title="e%s" % i,
                    content_object=Link.objects.create(
                        url="#%s" % i,
                    ),
                    enabled=True,
                )
            )

        adminurl = reverse("admin:menus_menu_change", args=[menu.pk])
        data = self._data_from_menu(menu)
        self.client.post(adminurl, data)
        self.assertEqual([entry["entry_title"] for entry in menu.children_list()], ["e1", "e2"])
        data.update(
            {
                "entry-order": "[%s]=null&[%s]=null" % (entries[1].pk, entries[0].pk),
            }
        )
        self.client.post(adminurl, data)
        self.assertEqual([e["entry_title"] for e in menu.children_list()], ["e2", "e1"])

    def test_addentry(self):
        menu = Menu.objects.add_root(title="Menu1", enabled=True)
        entries = []
        for i in range(1, 3):
            entries.append(
                MenuEntry.objects.add_child(
                    parent=menu,
                    title="e%s" % i,
                    content_object=Link.objects.create(
                        url="#%s" % i,
                    ),
                    enabled=True,
                )
            )
            adminurl = reverse("admin:menus_menu_change", args=[menu.pk])
            data = self._data_from_menu(menu)
            data["addentry"] = "menus#link"

        result = self.client.post(adminurl, data)
        self.assertEqual(result.status_code, 302)

    def test_addableentries(self):
        admininstance = MenuAdmin(model=Menu, admin_site=self.site)

        menu = Menu.objects.add_root(title="Menu1")
        entries = admininstance.addable_entries(obj=menu)
        len_e = len(MenusConfig.entrieable_models)
        if not TEMPLATETAG_WIDGETS:
            len_e -= 1
        self.assertEqual(entries.count("<li>"), len_e)

    def test_mk_entriableadmin(self):
        admincls = CPMAdmin
        make_entrieable_admin(admincls)
        self.assertTrue(admincls.is_menu_entrieable)

        instance = ConcretePublishableLangModel.objects.create(foo="x")
        data = instance.__dict__
        menu = Menu.objects.add_root(title="Menu1")

        admininstance = admincls(model=ConcretePublishableLangModel, admin_site=self.site)
        self.assertTrue(admininstance.get_fields({}, instance), [])
        self.assertTrue(admininstance.get_fieldsets({}, instance), [])
        formcls = admincls.form
        formcls._meta.model = ConcretePublishableLangModel
        form = formcls(instance.__dict__)
        self.assertTrue("menus" in form.fields.keys())
        valid = form.is_valid()
        self.assertTrue(valid)

        data["menus"] = [menu.pk]
        form = formcls(data, instance=instance)
        self.assertTrue("menus" in form.fields.keys())
        valid = form.is_valid()
        self.assertTrue(valid)

        data["status_changed_0"] = "2016-01-01"
        data["status_changed_1"] = "23:00"

        adminurl = get_adminedit_url(instance)
        response = self.client.post(adminurl, data)
        self.assertIn(response.status_code, (200, 302))
        self.assertEqual(MenuEntry.objects.count(), 2)
        response = self.client.get(adminurl)
        content = str(response.content)
        start = content.find('<input type="checkbox" name="menus"')
        end = content[start:].find(">")
        tag = content[start : start + end + 1]
        self.assertTrue(" checked" in tag)
