from jproperties import Properties

class ValidatorUtil:
    """
        Create a validator with the property configuration
        Functions :
        : validate_tomcat : Validate the tomcat server configurations
        : validate_property : Validate local.properties and localextensions.xml
    """

    @staticmethod
    def get_properties(path):
        local_properties = Properties()
        with open(path if path.endswith('/') else path+'/' +'local.properties','rb') as local_prop:
            local_properties.load(local_prop)
        return local_properties

    def __init__(self,property_url):
        self.properties = self.get_properties(property_url)
        self.
        
    def validate_tomcat(self):
        pass

    def validate_property(self):
        pass