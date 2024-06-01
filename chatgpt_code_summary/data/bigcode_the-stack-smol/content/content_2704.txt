### Package Import ###
from bson import ObjectId
from pydantic import BaseModel
from pydantic import fields
from pydantic.fields import Field
from typing import Optional
### AppCode Import ###
from Server.Model.POID import PyObjectId

###############################################################################

class User(BaseModel):
    Id: PyObjectId = Field(default_factory=PyObjectId, alias='_id')
    FirstName: str = Field(alias='FirstName')
    LastName: str = Field(alias='LastName')
    Email: str = Field(alias='Email')
    PhoneNumber: str = Field(alias='PhoneNumber')
    Password: str = Field(alias='Password')
    About: Optional[str] = Field(alias = 'About')
    ProfileUrl: Optional[str] = Field(alias='ProfileUrl')

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "FirstName": "Jane",
                "LastName": "Doe",
                "Email": "jdoe@example.com",
                "PhoneNumber": "6285588974456",
                "Password": "jdoee"
            }
        }

###############################################################################

class UserUpdateModel(BaseModel):
    FirstName: Optional[str] = Field(alias ='FirstName')
    LastName: Optional[str] = Field(alias='LastName')
    Email: Optional[str] = Field(alias='Email')
    PhoneNumber: Optional[str] = Field(alias='PhoneNumber')
    Password: Optional[str] = Field(alias='Password')
    About: Optional[str] = Field(alias = 'About')
    ProfileUrl: Optional[str] = Field(alias='ProfileUrl')

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "FirstName": "Jane",
                "LastName": "Doe",
                "Email": "jdoe@example.com",
                "PhoneNumber": "6285588974456",
                "Password": "jdoee",
                "About": "About jane doe",
                "ProfileUrl": "https://profileurlembed.com/file/janedoe"
            }
        }

###############################################################################