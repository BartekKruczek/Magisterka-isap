import time
import os
import pandas as pd
import warnings

from data import Data
from utils import Utils
from qwen2 import Qwen2

def main():
    warnings.filterwarnings("ignore")
    start_time = time.time()

    # auto debug kernel
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    data = Data(json_path='lemkin-json-from-html', pdf_path=  'lemkin-pdf')
    utils = Utils(json_path='lemkin-json-from-html', pdf_path = 'lemkin-pdf')
    qwen2 = Qwen2()

    # how many files are there in both directories
    # print("Detected {} .json and {} .pdf files".format(data.number_of_files()[0], data.number_of_files()[1]))

    years_to_convert: list[int] = [2015]
    # converting pdf to png
    once_converted = False
    if once_converted:
        for i in years_to_convert:
            for pdf_path in data.yield_pdf_files(year = i):
                print(utils.convert_pdf_to_png(pdf_path, year = i))

    years = [2014]

    # extract text from png
    # image_path = './lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_0.png'
    # text = data.get_text_from_png(image_path)

    # clean up text and combine it to one string, combined text is one string without spaces
    # cleaned_text = data.clean_text(text)
    # combined = data.combine_text_to_one_string(cleaned_text)
    # print(f"Pdf text \n{combined}")

    # yield json files, updated
    do_excel_json = False
    if do_excel_json:
        extracted_data = []
        skipped_files = []
        skipped_count = 0
        extracted_count = 0

        for year in years:
            for elem in utils.yield_json_files(year=year):
                json_data = data.read_json_data(elem)
                json_text = data.clean_text_from_json(data.get_text_from_json(json_data))
                json_text_data = data.get_text_data(text = json_text, only_first_date = True)
                formatted_date = json_text_data if json_text_data else "No date found"

                if formatted_date == "No date found":
                    skipped_files.append((elem, json_text))
                    skipped_count += 1
                else:
                    extracted_data.append((elem, formatted_date, json_text))
                    extracted_count += 1

                print(f"Json file: {elem}, extracted date: {formatted_date}")


        df = pd.DataFrame(extracted_data, columns=["JSON file path", "Extracted Date", "Text_json"])
        df.to_excel("extracted_json_dates.xlsx", index=False, engine="openpyxl")
        print("Data saved to extracted_json_dates.xlsx")
        print(f"Extracted_count: {extracted_count}")

        if skipped_files:
            skipped_df = pd.DataFrame(skipped_files, columns=["Skipped JSON file path", "Text_json"])
            skipped_df.to_excel("no_dates_found_json.xlsx", index = False)
            print(f"Skipped files saved to no_dates_found_json.xlsx with {skipped_count} entries.")

    # read json data
    # json_data = data.read_json_data('lemkin-json-from-html/2014/2014_1594.json')
    # json_text = data.clean_text_from_json(data.get_text_from_json(json_data))
    # print(f"Json text \n{json_text}")

    # perform LCS, only one pdf file and one json file
    # print(utils.longest_common_subsequence_dynamic(utils.list_of_json_paths(), combined))

    # json text debugger
    # utils.json_text_debugger(iterator = utils.yield_json_files(), my_data = data)

    # png_list_debugger
    # utils.pngs_list_debugger(pngs_list, data)

    # data.delete_unwanted_folders()

    do_debug_png_folders = False
    if do_debug_png_folders:
        path = './lemkin-pdf/2014/WDU20140000693/O/D20140693_png'
        text = data.get_text_from_images(image_folder = path)

        # data
        date = data.get_text_data(text)
        formated_date = date.strftime('%Y-%m-%d')
        print(f'Formatted date: {formated_date}')

    do_folder_debug = False
    if do_folder_debug:
        for year in years:
            image_folders = []
            base_path = os.path.join(data.pdf_path, str(year))

            for root, dirs, _ in os.walk(base_path):
                for dir_name in dirs:
                    for root2, dirs2, _ in os.walk(os.path.join(root, dir_name)):
                        for dir2 in dirs2:
                            if dir2.endswith('_png'):
                                image_folders.append(os.path.join(root2, dir2))

        print(image_folders)

    do_debug_combine = False
    if do_debug_combine:
        image_folders = []
        for year in years:
            base_path = os.path.join(data.pdf_path, str(year))
            for root, dirs, _ in os.walk(base_path):
                for dir_name in dirs:
                    for root2, dirs2, _ in os.walk(os.path.join(root, dir_name)):
                        for dir2 in dirs2:
                            if dir2.endswith('_png'):
                                image_folders.append(os.path.join(root2, dir2))

        # print(image_folders)

        extracted_data = []
        skipped_files = []
        skipped_count = 0
        extracted_count = 0
        for elem in image_folders:
            text = data.get_text_from_images(image_folder = elem, first_png_only = True)

            # data dokumentu, dla pdf-ów chcemy drugą datę
            date = data.get_text_data(text = text, only_first_date = False)

            if date is not None:
                formatted_date = date if date else "No date found"
                print(f'Formatted date: {formatted_date}, image folder path: {elem}')
                extracted_data.append((elem, formatted_date, text))
                extracted_count += 1
            else:
                print("No date found in the text.")
                extracted_data.append((elem, "No date found", text))

                skipped_files.append((elem, text))
                skipped_count += 1

        df = pd.DataFrame(extracted_data, columns=["Image folder path", "Extracted Date", "Text"])
        df.to_excel("extracted_dates.xlsx", index = False, engine = "openpyxl")
        print("Data saved to extracted_dates.xlsx")
        print(f"Extracted counter: {extracted_count}")

        if skipped_files:
            skipped_df = pd.DataFrame(skipped_files, columns=["Skipped pdf file path", "Text"])
            skipped_df.to_excel("no_dates_found_pdf.xlsx", index = False, engine = "openpyxl")
            print(f"Skipped files saved to no_dates_found_pdf.xlsx with {skipped_count} entries.")

    do_iterate = False
    if do_iterate:
        for year in years:
            image_folders = []
            base_path = os.path.join(data.pdf_path, str(year))

            for root, dirs, _ in os.walk(base_path):
                for dir_name in dirs:
                    for root2, dirs2, _ in os.walk(os.path.join(root, dir_name)):
                        for dir2 in dirs2:
                            if dir2.endswith('_png'):
                                image_folders.append(os.path.join(root2, dir2))

            # utils.find_max_lcs(utils.yield_json_files(year = year), utils.png_paths_creator(year), data, year)
            utils.find_max_lcs_folders(utils.yield_json_files(year=year), image_folders, data, year)

    # utils.find_matching_dates()
    # utils.calculate_cosine_similarity()
    # utils.check_similarities()
    # utils.spacy_tester()
    # utils.find_start_end_each_page()

    # DO NOT TOUCH!!!
    data.clean_xlsx()
    data.create_new_xlsx()

    # json section
    qwen2.create_json()

    # auto repair json section
    # qwen2.auto_repair_json_QWEN()

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Czas wykonania: {elapsed_time:.2f} minut")

if __name__ == '__main__':
    main()
