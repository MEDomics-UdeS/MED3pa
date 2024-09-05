"""
Defines a function to convert the results of an experiment into a Med3paResult file. The created Med3paResult file can
then be used to visualize the experiment in the MEDomicsLab app (https://medomics-udes.gitbook.io/medomicslab-docs).
"""


import os
import datetime
import json
from pathlib import Path


def generate_Med3paResults(path_to_results):

    file_name = f"/MED3paResults_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}".replace(
        r'[^a-zA-Z0-9-_]', "")
    is_detectron = False
    tabs = ["infoConfig", "reference", "test"]

    file_content = {"loadedFiles": {}, "isDetectron": False}
    file_path = ""

    if path_to_results.startswith("detectron"):
        is_detectron = True
        file_path = os.path.join(path_to_results, "detectron_results")
        load_and_handle_files(file_path, file_content, None)
    else:
        for tab in tabs:
            file_path = ""
            if tab == "reference":
                file_path = os.path.join(path_to_results, "reference")
            elif tab == "test":
                if path_to_results.startswith("med3pa_detectron"):
                    file_path = os.path.join(path_to_results, "test")
                    load_and_handle_files(file_path, file_content, "test")
                    load_and_handle_files(os.path.join(path_to_results, "detectron"), file_content,
                                          "detectron_results")
                elif path_to_results.startswith("med3"):
                    file_path = os.path.join(path_to_results, "test")
                    load_and_handle_files(file_path, file_content, "test")
            else:
                file_path = path_to_results
            if file_path:
                load_and_handle_files(file_path, file_content, tab)

    file_content["isDetectron"] = is_detectron
    save_dict_to_file(file_content, path_to_results + file_name + '.MED3paResults')


def get_all_text_in_folder(folder_path):
    all_text = {}

    # List all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            try:
                if file_name.endswith('.json'):
                    # Open the JSON file and load its content as a dictionary
                    with open(file_path, 'r', encoding='utf-8') as file:
                        all_text[Path(file_name).stem] = json.load(file)
                # else:
                #     # Open the file and read its content as text
                #     with open(file_path, 'r', encoding='utf-8') as file:
                #         all_text[Path(file_name).stem] = file.read()

                # # Open the file and read its content
                # with open(file_path, 'r', encoding='utf-8') as file:
                #     all_text[file_name] = file.read()
                #     # all_text += file.read()  # + "\n"
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return all_text


def load_and_handle_files(file_path, file_content, tab):
    try:
        # files = load_json_files(file_path)
        files = get_all_text_in_folder(file_path)
        if tab is None:
            file_content['loadedFiles'] = files
        else:
            file_content['loadedFiles'][tab] = files
    except Exception as error:
        print(f"Error loading {tab} files:", error)


def to_string(value):
    if value is None:
        return 'null'
    return str(value)


def write_list(file, l, indent, inc):
    if len(l) == 0:
        file.write("[]")
    else:
        file.write('[\n')
        for idx, item in enumerate(l):
            coma_list = '' if idx == len(l) - 1 else ','
            if isinstance(item, list):
                write_list(file, item, indent + inc, inc)
            elif isinstance(item, dict):
                file.write(" " * (indent + inc))
                write_dict(file, item, indent + inc, inc)
            else:
                file.write(' ' * (indent + inc) + '"' + to_string(item) + '"')
            file.write(coma_list + "\n")
        file.write(' ' * indent + ']')  # + coma + '\n'


def write_dict(file, d, indent=0, inc=2):
    # file.write("{")
    if len(d) == 0:
        file.write("{}")
    else:
        file.write("{\n")  # " " * indent +
        for index, (key, value) in enumerate(d.items()):
            file.write(f'{" " * (indent + inc)}"{key}": ')
            coma = '' if index == len(d) - 1 else ','
            if isinstance(value, dict):
                # if len(value) == 0:
                #     file.write("{}" + coma + "\n")
                # else:
                #     file.write("{\n")
                write_dict(file, value, (indent + inc), inc)
                file.write(coma + '\n')
                # file.write(" " * (indent + inc) + "}" + coma + "\n")

            elif isinstance(value, list):
                # file.write('[\n')
                write_list(file, value, (indent + inc), inc)
                # for idx, item in enumerate(value):
                #     coma_list = '' if idx == len(value) - 1 else ','
                #     file.write(' ' * (indent + 2) + "'" + to_string(item) + "'" + coma_list + '\n')
                # file.write(' ' * (indent + inc) + ']' + coma + '\n')
                file.write(coma + '\n')

            elif isinstance(value, bool):
                file.write(f'{str(value).lower()}{coma}\n')
            elif isinstance(value, str):
                file.write(f'"{value}"{coma}\n')
            else:
                file.write(f'{to_string(value)}{coma}\n')
        file.write(" " * indent + "}")


def save_dict_to_file(dictionary, file_path):
    # print(dictionary["loadedFiles"]["infoConfig"])

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            write_dict(file, dictionary)
            # file.write("}")

        print(f"Dictionary successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving dictionary to {file_path}: {e}")
