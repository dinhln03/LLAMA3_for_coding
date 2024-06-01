from hotel_app.views import *
from rest_framework import routers

router = routers.DefaultRouter()
router.register(r'rooms', RoomAPIView)
router.register(r'employee', EmployeeAPIView)
router.register(r'resident', ResidentAPIView)
router.register(r'booking', BookingRecordAPIView)
router.register(r'cleaning', CleaningScheduleAPIView)
