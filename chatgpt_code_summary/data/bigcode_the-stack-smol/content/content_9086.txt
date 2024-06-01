import requests
import json
import os
import copy
import smtplib
import jwt
from datetime import datetime, timedelta
# SMTP 라이브러리
from string import Template  # 문자열 템플릿 모듈
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from django.core.validators import validate_email, ValidationError
# email templete
from .templates import EMAIL_TEMPLATE
from django.http import HttpResponseRedirect
# my settings
from my_settings import (
    EMAIL_ADDRESS,
    EMAIL_PASSWORD,
    SECRET,
    ALGORITHM,
)
from django.shortcuts import redirect
from django.conf import settings
from django.shortcuts import redirect
from django.contrib.auth import get_user_model
from django.http import JsonResponse, HttpResponse
from django.db.models import Q
from .models import User, Corrector, Applicant
from .serializer import (
    ApplicantSerializer,
    UserSerializer,
    CorrectorSerializer,
)
# from rest_framework.views import APIView
from rest_framework import viewsets
from allauth.socialaccount.providers.kakao import views as kakao_views
from allauth.socialaccount.providers.kakao.views import KakaoOAuth2Adapter
from allauth.socialaccount.providers.oauth2.client import OAuth2Client
from rest_auth.registration.views import SocialLoginView
from rest_framework.decorators import (
    api_view,
    permission_classes,
    authentication_classes,
)
from rest_framework.permissions import IsAuthenticated
from rest_framework_jwt.authentication import JSONWebTokenAuthentication
ip = "192.168.0.137"
class ApplicantView(viewsets.ModelViewSet):
    queryset = Applicant.objects.all()
    serializer_class = ApplicantSerializer
class UserView(viewsets.ViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
User = get_user_model()
def kakaoSignup(request):
    # def post(self, request):
    access_token = request.headers["Authorization"]
    profile_request = requests.post(
        "https://kapi.kakao.com/v2/user/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    profile_json = profile_request.json()
    kakao_id = profile_request.json().get("id")
    print("profile_json : ", profile_json)
    kakao_account = profile_json.get("kakao_account")
    print("kakao_account : ", kakao_account)
    email = kakao_account.get("email", None)
    profile = kakao_account.get("profile")
    print("profile : ", profile)
    nickname = profile.get("nickname")
    profile_image = profile.get("profile_image_url")
    data = {"access_token": access_token}
    accept = requests.post("http://localhost:8000/accounts/rest-auth/kakao/", data=data)
    accept_json = accept.json()
    print(accept_json)
    accept_jwt = accept_json.get("token")
    try:
        user = User.objects.get(kakao_id=kakao_id)
        return JsonResponse(
            {
                "access_token": accept_jwt,
                "name": nickname,
                "image": profile_image,
                "is_corrector": user.is_corrector,
                "is_phone_auth": user.is_phone_auth,
            },
            status=200,
        )
    # redirect('http://localhost:3000/')
    except User.DoesNotExist:
        User.objects.filter(email=email).update(
            kakao_id=kakao_id, name=nickname, thumbnail_image=profile_image,
        )
        user = User.objects.get(kakao_id=kakao_id)
        return JsonResponse(
            {
                "access_token": accept_jwt,
                "name": nickname,
                "image": profile_image,
                "is_corrector": user.is_corrector,
                "is_phone_auth": user.is_phone_auth,
            },
            status=200,
        )
class KakaoLoginView(SocialLoginView):
    adapter_class = kakao_views.KakaoOAuth2Adapter
    client_class = OAuth2Client
class EmailHTMLContent:
    """e메일에 담길 컨텐츠"""
    def __init__(self, str_subject, template, template_params):
        """string template과 딕셔너리형 template_params받아 MIME 메시지를 만든다"""
        assert isinstance(template, Template)
        assert isinstance(template_params, dict)
        self.msg = MIMEMultipart()
        # e메일 제목을 설정한다
        self.msg["Subject"] = str_subject  # e메일 제목을 설정한다
        # e메일 본문을 설정한다
        str_msg = template.safe_substitute(**template_params)  # ${변수} 치환하며 문자열 만든다
        # MIME HTML 문자열을 만든다
        mime_msg = MIMEText(str_msg, "html")
        self.msg.attach(mime_msg)
    def get_message(self, str_from_email_addr, str_to_email_addr):
        """발신자, 수신자리스트를 이용하여 보낼메시지를 만든다 """
        send_msg = copy.deepcopy(self.msg)
        send_msg["From"] = str_from_email_addr  # 발신자
        # ",".join(str_to_email_addrs) : 수신자리스트 2개 이상인 경우
        send_msg["To"] = str_to_email_addr
        return send_msg
class EmailSender:
    """e메일 발송자"""
    def __init__(self, str_host, num_port):
        """호스트와 포트번호로 SMTP로 연결한다 """
        self.str_host = str_host
        self.num_port = num_port
        self.smtp_connect = smtplib.SMTP(host=str_host, port=num_port)
        # SMTP인증이 필요하면 아래 주석을 해제하세요.
        self.smtp_connect.starttls()  # TLS(Transport Layer Security) 시작
        self.smtp_connect.login(EMAIL_ADDRESS, EMAIL_PASSWORD)  # 메일서버에 연결한 계정과 비밀번호
    def send_message(self, emailContent, str_from_email_addr, str_to_email_addr):
        """e메일을 발송한다 """
        contents = emailContent.get_message(str_from_email_addr, str_to_email_addr)
        self.smtp_connect.send_message(
            contents, from_addr=str_from_email_addr, to_addrs=str_to_email_addr
        )
        del contents
class EmailAuthenticationView(viewsets.ModelViewSet):
    _COMMON_EMAIL_ADDRESS = [
        "hanmail.net",
        "hotmail.com",
        "yahoo.co.kr",
        "yahoo.com",
        "hanmir.com",
        "nate.com",
        "dreamwiz.com",
        "freechal.com",
        "teramail.com",
        "metq.com",
        "lycos.co.kr",
        "chol.com",
        "korea.com",
        ".edu",
        ".ac.kr",
        # "naver.com"
    ]
    serializer_class = CorrectorSerializer
    @permission_classes((IsAuthenticated,))
    @authentication_classes((JSONWebTokenAuthentication,))
    @classmethod
    def create(cls, request, *args, **kwargs):
        try:
            company_email = request.data["email"]  # company email
            user_email = request.user.email
            if User.objects.filter(email=user_email, is_corrector=True).exists():
                return JsonResponse({"message": "ALREADY AUTHENTICATED"}, status=406)
            user = User.objects.get(email=user_email)
            try:
                # email validator
                validate_email(company_email)
                Corrector(user=user, company_email=company_email).save()
            except ValidationError:
                return JsonResponse({"message": "INVALID EMAIL"}, status=400)
            except PermissionError:
                return JsonResponse({"message": "PERMISSION ERROR"}, status=401)
            if company_email.split("@")[1] in cls._COMMON_EMAIL_ADDRESS:
                return JsonResponse(
                    {"message": "NOT COMPANY EMAIL ADDRESS"}, status=400
                )
            str_host = "smtp.gmail.com"
            num_port = 587  # SMTP Port
            emailSender = EmailSender(str_host, num_port)
            str_subject = "[픽소서] EMAIL 인증을 완료해주세요!"  # e메일 제목
            auth_token = jwt.encode(
                {"email": company_email, "name": user.name},
                SECRET["secret"],
                algorithm=ALGORITHM,
            ).decode("utf-8")
            template = Template(EMAIL_TEMPLATE)
            template_params = {
                "From": EMAIL_ADDRESS,
                "Token": auth_token,
                "BackendIp": ip,
            }
            emailHTMLContent = EmailHTMLContent(str_subject, template, template_params)
            str_from_email_addr = EMAIL_ADDRESS  # 발신자
            str_to_email_addr = company_email  # 수신자/ 2개 이상인 경우 리스트
            emailSender.send_message(
                emailHTMLContent, str_from_email_addr, str_to_email_addr
            )
            User.objects.filter(email=user.email).update(email_auth_token=auth_token)
            return JsonResponse(
                {"message": "EMAIL SENT", "EMAIL_AUTH_TOKEN": auth_token}, status=200
            )
        except KeyError as e:
            return JsonResponse({"message": f"KEY ERROR {e}"}, status=400)
class EmailAuthSuccView(viewsets.ModelViewSet):
    serializer_class = UserSerializer
    queryset = User.objects.all()
    def create(self, request, *args, **kwargs):
        auth_token = self.kwargs["auth_token"]
        user_queryset = self.queryset.filter(email_auth_token=auth_token)
        try:
            if user_queryset.exists():
                user = User.objects.get(
                    email_auth_token=user_queryset.first().email_auth_token
                )
                user.is_corrector = True
                user.save()
                return HttpResponseRedirect(f"http://localhost:3000/")
            return JsonResponse({"message": "USER DOES NOT EXIST"}, status=400)
        except jwt.exceptions.DecodeError:
            return JsonResponse({"message": "INVALID TOKEN"}, status=400)
        except jwt.exceptions.ExpiredSignatureError:
            return JsonResponse({"message": "EXPIRED TOKEN"}, status=400)
