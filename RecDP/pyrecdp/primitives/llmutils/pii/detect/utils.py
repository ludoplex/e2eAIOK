from enum import Enum, auto


class PIIEntityType(Enum):
    IP_ADDRESS = auto()
    NAME = auto()
    EMAIL = auto()
    PHONE_NUMBER = auto()
    PASSWORD = auto()
    KEY = auto()

    @classmethod
    def default(cls):
        return [PIIEntityType.IP_ADDRESS, PIIEntityType.EMAIL, PIIEntityType.PHONE_NUMBER, PIIEntityType.KEY]

    @classmethod
    def parse(cls, entity):
        if entity == "name":
            return PIIEntityType.NAME
        elif entity == "password":
            return PIIEntityType.PASSWORD
        elif entity == "email":
            return PIIEntityType.EMAIL
        elif entity == "phone_number":
            return PIIEntityType.PHONE_NUMBER
        elif entity == "ip":
            return PIIEntityType.PHONE_NUMBER
        elif entity == "key":
            return PIIEntityType.KEY
        else:
            raise NotImplementedError(f" entity type {entity} is not supported!")
