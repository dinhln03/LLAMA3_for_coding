from mitsubishi_central_controller.util.ControllerDictBuilder import ControllerDictBuilder
import aiohttp
import asyncio

from mitsubishi_central_controller.util.dict_utils import get_group_list_from_dict, get_system_data_from_dict, \
    get_single_bulk_from_dict, get_single_racsw_from_dict, get_single_energycontrol_from_dict, get_lcd_name_from_dict, \
    get_group_info_list_from_dict
from mitsubishi_central_controller.util.temperature_utils import f_to_c
from mitsubishi_central_controller.util.xml_utils import parse_xml


class CentralController:
    def __init__(self, url):
        self.url = url
        self.full_url = url + "/servlet/MIMEReceiveServlet"
        self.session = None
        self.groups = None
        self.system_data = None
        self.semaphore = None

    def print(self):
        print(self.__dict__)

    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
            self.semaphore = asyncio.Semaphore(value=7)
            return self.session
        else:
            return self.session

    async def initialize_group(self, group):
        await self.async_update_single_group_bulk(group)
        group.update_from_bulk()
        print(group.__dict__)

    async def initialize_all(self):
        await self.async_initialize_system_data()
        await self.async_initialize_group_list()
        await asyncio.wait([self.initialize_group(group) for group in self.groups])

    async def async_send_command(self, command):
        session = await self.get_session()
        await self.semaphore.acquire()
        resp = await session.post(self.full_url, data=command, headers={'Content-Type': 'text/xml'})
        self.semaphore.release()
        return await resp.text()

    async def async_initialize_system_data(self):
        xml = ControllerDictBuilder().get_system_data().to_xml()
        xml_response = await self.async_send_command(xml)
        parsed = parse_xml(xml_response)
        self.system_data = get_system_data_from_dict(parsed)

    async def async_initialize_group_list(self):
        xml = ControllerDictBuilder().get_mnet_group_list().to_xml()
        xml_response = await self.async_send_command(xml)
        parsed = parse_xml(xml_response)
        self.groups = get_group_list_from_dict(parsed)
        await self.async_update_group_list_with_names()

    async def async_update_group_list_with_names(self):
        xml = ControllerDictBuilder().get_mnet_list().to_xml()
        xml_response = await self.async_send_command(xml)
        parsed = parse_xml(xml_response)
        groups_info = get_group_info_list_from_dict(parsed)
        for group in self.groups:
            group.web_name = groups_info[group.group_id]["web_name"]
            group.lcd_name = groups_info[group.group_id]["lcd_name"]

    async def async_update_single_group_bulk(self, group):
        xml = ControllerDictBuilder().get_single_bulk_data(group.group_id).to_xml()
        xml_response = await self.async_send_command(xml)
        parsed = parse_xml(xml_response)
        group.bulk_string = get_single_bulk_from_dict(parsed)
        group.rac_sw = get_single_racsw_from_dict(parsed)
        group.energy_control = get_single_energycontrol_from_dict(parsed)
        return group

    async def update_lcd_name_for_group(self, group):
        xml = ControllerDictBuilder().get_mnet(group.group_id, lcd_name=True).to_xml()
        xml_response = await self.async_send_command(xml)
        parsed = parse_xml(xml_response)
        group.lcd_name = get_lcd_name_from_dict(parsed)

    async def set_drive_for_group(self, group, drive_string):
        xml = ControllerDictBuilder().set_mnet(group.group_id, drive=drive_string).to_xml()
        await self.async_send_command(xml)
        await self.async_update_single_group_bulk(group)

    async def set_mode_for_group(self, group, mode):
        xml = ControllerDictBuilder().set_mnet(group.group_id, mode=mode).to_xml()
        await self.async_send_command(xml)
        await self.async_update_single_group_bulk(group)

    async def set_temperature_fahrenheit_for_group(self, group, temperature):
        xml = ControllerDictBuilder().set_mnet(group.group_id, set_temp=f_to_c(int(temperature))).to_xml()
        await self.async_send_command(xml)
        await self.async_update_single_group_bulk(group)

    async def set_air_direction_for_group(self, group, air_direction):
        xml = ControllerDictBuilder().set_mnet(group.group_id, air_direction=air_direction).to_xml()
        await self.async_send_command(xml)
        await self.async_update_single_group_bulk(group)

    async def set_fan_speed_for_group(self, group, fan_speed):
        xml = ControllerDictBuilder().set_mnet(group.group_id, fan_speed=fan_speed).to_xml()
        await self.async_send_command(xml)
        await self.async_update_single_group_bulk(group)

    async def set_remote_controller_for_group(self, group, remote_controller):
        xml = ControllerDictBuilder().set_mnet(group.group_id, remote_controller=remote_controller).to_xml()
        await self.async_send_command(xml)
        await self.async_update_single_group_bulk(group)

    async def reset_filter_for_group(self, group):
        xml = ControllerDictBuilder().set_mnet(group.group_id, filter_sign="RESET").to_xml()
        await self.async_send_command(xml)
        await self.async_update_single_group_bulk(group)

    async def reset_error_for_group(self, group):
        xml = ControllerDictBuilder().set_mnet(group.group_id, error_sign="RESET").to_xml()
        await self.async_send_command(xml)
        await self.async_update_single_group_bulk(group)

    async def close_connection(self):
        s = await self.get_session()
        await s.close()
