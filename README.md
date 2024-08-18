# Plugin for using tesseract with ocr_translate

This is a plugin for the [ocr_translate](https://github.com/Crivella/ocr_translate) server for implementing translations using [tesseract](https://tesseract-ocr.github.io/)

**IMPORANT**: This plugin required tesseract to be installed to work [link](https://tesseract-ocr.github.io/tessdoc/Installation.html)

## Usage

For versions of the server `<v0.6`

- Install this by running `pip install ocr_translate-tesseract`
- Add `ocr_translate_tesseract` to your `INSTALLED_APPS` in `settings.py`
- Run the server with `AUTOCREATE_VALIDATED_MODELS` once

For versions of the server `>=0.6`

- Install through the server plugin manager
