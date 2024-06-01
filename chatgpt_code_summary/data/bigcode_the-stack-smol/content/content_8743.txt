from apollo.viewsets import UserViewSet
from applications.assets.viewsets import EquipmentViewSet, ServiceViewSet
from applications.business.viewsets import BusinessViewSet, BusinessMembershipViewSet
from applications.charge_list.viewsets import ChargeListViewSet, ActivityChargeViewSet, \
    ActivityChargeActivityCountViewSet, TimeChargeViewSet, UnitChargeViewSet
from applications.price_list.viewsets import PriceListViewSet, ActivityPriceListItemViewSet, TimePriceListItemViewSet, \
    UnitPriceListItemViewSet, PriceListItemEquipmentViewSet, PriceListItemServiceViewSet
from applications.station.viewsets import StationViewSet, StationBusinessViewSet, StationRentalViewSet
from applications.terms_of_service.viewsets import TermsOfServiceViewSet
from rest_framework.routers import DefaultRouter

# Internal API Definition
router = DefaultRouter()

router.register(r'account/user', UserViewSet, base_name='user')
router.register(r'account/terms_of_service', TermsOfServiceViewSet, base_name='terms-of-service')
router.register(r'business/business', BusinessViewSet, base_name='business')
router.register(r'business/business_membership', BusinessMembershipViewSet, base_name='business-membership')
router.register(r'equipment/equipment', EquipmentViewSet, base_name='equipment')
router.register(r'equipment/service', ServiceViewSet, base_name='service')
router.register(r'station/station', StationViewSet, base_name='station')
router.register(r'station/station_business', StationBusinessViewSet, base_name='station-business')
router.register(r'station/station_rental', StationRentalViewSet, base_name='station-rental')
router.register(r'price_list/price_list', PriceListViewSet, base_name='price-list')
router.register(r'price_list/activity_item', ActivityPriceListItemViewSet, base_name='activity-price-list-item')
router.register(r'price_list/time_item', TimePriceListItemViewSet, base_name='time-price-list-item')
router.register(r'price_list/unit_item', UnitPriceListItemViewSet, base_name='unit-price-list-item')
router.register(r'price_list/equipment_relation', PriceListItemEquipmentViewSet, base_name='price-list-item-equipment')
router.register(r'price_list/service_relation', PriceListItemServiceViewSet, base_name='price-list-item-service')
router.register(r'charge_list/charge_list', ChargeListViewSet, base_name='charge-list')
router.register(r'charge_list/activity_charge', ActivityChargeViewSet, base_name='activity-charge')
router.register(r'charge_list/activity_charge_activity_count', ActivityChargeActivityCountViewSet,
                base_name='activity-charge-activity-count')
router.register(r'charge_list/time_charge', TimeChargeViewSet, base_name='time-charge')
router.register(r'charge_list/unit_charge', UnitChargeViewSet, base_name='unit-charge')

