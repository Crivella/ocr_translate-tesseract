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
"""Tests for ocr_translate_tesseract plugin."""

import pytest

import ocr_translate_tesseract.plugin as octt_plugin


@pytest.fixture()
def tesseract_model_dict():
    """OCRModel dictionary."""
    return {
        'name': 'tesseract',
        'language_format': 'iso1',
        'entrypoint': 'tesseract.ocr'
    }

@pytest.fixture()
def tesseract_model(tesseract_model_dict):
    """OCRModel database object."""

    return octt_plugin.TesseractOCRModel(**tesseract_model_dict)

@pytest.fixture(scope='function')
def mock_called(request):
    """Generic mock function to check if it was called."""
    def mock_call(*args, **kwargs): # pylint: disable=inconsistent-return-statements
        mock_call.called = True
        mock_call.args = args
        mock_call.kwargs = kwargs

        if hasattr(request, 'param'):
            return request.param

    if hasattr(request, 'param'):
        mock_call.expected = request.param

    return mock_call
