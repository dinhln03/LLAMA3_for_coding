# Copyright 2022 Aleksandr Soloshenko
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from django.conf import settings
from storages.backends.azure_storage import AzureStorage

# Create your models here.
class PhotoStorage(AzureStorage):
    account_name = os.getenv("AZURE_ACCOUNT_NAME")
    account_key = os.getenv("AZURE_ACCOUNT_KEY")
    azure_container = os.getenv("AZURE_CONTAINER")
    expiration_secs = None


photoStorage = PhotoStorage()
