import phonenumbers


def detect_phones(text):
    """Detects phone in a string using phonenumbers libray only detection the international phone number"""
    return [
        {
            "tag": "PHONE_NUMBER",
            "value": match.raw_string,
            "start": match.start,
            "end": match.end,
        }
        for match in phonenumbers.PhoneNumberMatcher(text, "IN")
    ]
