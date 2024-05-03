###################################################################################
# ocr_translate-tesseract - a tesseract plugin for ocr_translate                  #
# Copyright (C) 2023-present Davide Grassano                                      #
#                                                                                 #
# This program is free software: you can redistribute it and/or modify            #
# it under the terms of the GNU General Public License as published by            #
# the Free Software Foundation, either version 3 of the License.                  #
#                                                                                 #
# This program is distributed in the hope that it will be useful,                 #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                  #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                   #
# GNU General Public License for more details.                                    #
#                                                                                 #
# You should have received a copy of the GNU General Public License               #
# along with this program.  If not, see {http://www.gnu.org/licenses/}.           #
#                                                                                 #
# Home: https://github.com/Crivella/ocr_translate-tesseract                       #
###################################################################################
"""Plugin to implement tesseract OCR for ocr_translate."""

__version__ = '0.2.1'

tesseract_ocr_model_data = {
    'name': 'tesseract',
    'lang': [
        'af', 'sq', 'am', 'ar', 'hy', 'as', 'az', 'eu', 'be', 'bn', 'bs', 'br', 'bg', 'my', 'zh', 'zht', 'co', 'hr',
        'cs', 'da', 'dz', 'en', 'eo', 'et', 'fo', 'fi', 'fr', 'fy', 'gl', 'ka', 'de', 'gu', 'he', 'hi', 'hu', 'is',
        'id', 'iu', 'ga', 'it', 'ja', 'jv', 'kn', 'kk', 'km', 'ko', 'lo', 'la', 'lv', 'lt', 'mk', 'ms', 'ml', 'mt',
        'mi', 'mr', 'mn', 'ne', 'no', 'or', 'fa', 'pl', 'pt', 'qu', 'ru', 'sa', 'sr', 'sd', 'sk', 'sl', 'su', 'sw',
        'sv', 'tg', 'ta', 'tt', 'te', 'th', 'bo', 'ti', 'tr', 'uk', 'ur', 'uz', 'vi', 'cy', 'yi', 'yo'
        ],
    'lang_code': 'iso3',
    'entrypoint': 'tesseract.ocr',
    'iso1_map': {
        'zh': 'chi_sim',
        'zht': 'chi_tra',
        'et': 'est',
        'iu': 'iku',
        'lv': 'lav',
        'ms': 'msa',
        'mn': 'mon',
        'ne': 'nep',
        'no': 'nor',
        'or': 'ori',
        'fa': 'fas',
        'qu': 'que',
        'sw': 'swa',
        'uz': 'uzb',
        'yi': 'yid',
    }
}
