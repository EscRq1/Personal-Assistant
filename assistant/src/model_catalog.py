"""Selectable router aliases from the same preset consumed by llama.cpp."""
import configparser


def read_models(path):
    config = configparser.ConfigParser(interpolation=None, strict=True)
    with open(path, encoding="utf-8") as source:
        config.read_file(source)
    aliases = [section for section in config.sections()
               if section != "*" and not section.startswith("llama.")]
    if "observer" not in aliases:
        raise ValueError("models.ini must contain an [observer] section")
    return aliases
