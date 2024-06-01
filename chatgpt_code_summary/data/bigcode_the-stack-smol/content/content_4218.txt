# Copyright 2018 ZTE Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rest_framework import serializers
from .checksum import ChecksumSerializer


class VnfPackageSoftwareImageInfoSerializer(serializers.Serializer):
    id = serializers.CharField(
        help_text="Identifier of the software image.",
        required=True,
        allow_null=False,
        allow_blank=False
    )
    name = serializers.CharField(
        help_text="Name of the software image.",
        required=True,
        allow_null=True,
        allow_blank=False
    )
    provider = serializers.CharField(
        help_text="Provider of the software image.",
        required=True,
        allow_null=True,
        allow_blank=False
    )
    version = serializers.CharField(
        help_text="Version of the software image.",
        required=True,
        allow_null=True,
        allow_blank=False
    )
    checksum = ChecksumSerializer(
        help_text="Checksum of the software image file.",
        required=True,
        allow_null=False
    )
    containerFormat = serializers.ChoiceField(
        help_text="terminationType: Indicates whether forceful or graceful termination is requested.",
        choices=["AKI", "AMI", "ARI", "BARE", "DOCKER", "OVA", "OVF"],
        required=True,
        allow_null=True
    )
    diskFormat = serializers.ChoiceField(
        help_text="Disk format of a software image is the format of the underlying disk image.",
        choices=["AKI", "AMI", "ARI", "ISO", "QCOW2", "RAW", "VDI", "VHD", "VHDX", "VMDK"],
        required=True,
        allow_null=True
    )
    createdAt = serializers.DateTimeField(
        help_text="Time when this software image was created.",
        required=True,
        format=None,
        input_formats=None
    )
    minDisk = serializers.IntegerField(
        help_text="The minimal disk for this software image in bytes.",
        required=True,
        allow_null=True
    )
    minRam = serializers.IntegerField(
        help_text="The minimal RAM for this software image in bytes.",
        required=True,
        allow_null=True
    )
    size = serializers.IntegerField(
        help_text="Size of this software image in bytes.",
        required=True,
        allow_null=True
    )
    userMetadata = serializers.DictField(
        help_text="User-defined data.",
        child=serializers.CharField(
            help_text="KeyValue Pairs",
            allow_blank=True
        ),
        required=False,
        allow_null=True
    )
    imagePath = serializers.CharField(
        help_text="Path in the VNF package.",
        required=True,
        allow_null=True,
        allow_blank=False
    )
