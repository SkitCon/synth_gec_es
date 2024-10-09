'''
File: custom_errors.py
Author: Amber Converse
Purpose: Defines custom errors for SYNTH GEC ES project
'''

class InvalidLabelException(Exception):
    def __init__(self, message):
        super().__init__(message)

class NotInDictionaryException(Exception):
    def __init__(self, message):
        super().__init__(message)