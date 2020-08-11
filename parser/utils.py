import re
from .cts import *


def retrieve_email(text):
    return re.findall('\S+@\S+', text)


def retrieve_phone_number(text):
    pass


def retrieve_skills(text):
    return [skill for skill in skills if skill in text.lower()]


def retrieve_education_institution(text):
    pass


def retrieve_higher_degree(text):
    pass


def retrieve_dates(text):
    pass


def retrieve_past_experience(text):
    pass


def retrieve_last_experience_year(text):
    pass


def retrieve_name(text):
    pass


def parse_cv_sections(text):
    pass






