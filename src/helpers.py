from data_types.vectors import View
from globals import sign_operations


def select_operator(views: View):
    return sign_operations[views.sign_type]
