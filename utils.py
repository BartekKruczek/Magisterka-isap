import os
import pypdfium2 as pdfium
import pandas as pd
import pylcs
import spacy

from data import Data

class Utils(Data):
    def __init__(self, json_path: str, pdf_path: str) -> None:
        super().__init__(json_path, pdf_path)
        self.base_json_path = json_path
        self.base_pdf_path = pdf_path

    def __repr__(self) -> str:
        return "Klasa do obsługi różnych narzędzi"

    def json_folder_iterator(self):
        """
        Iterates over a directory with .json files
        """
        print(f'Initializing {self.json_folder_iterator.__name__}')
        for root, dirs, files in os.walk(self.json_path):
            for dir in dirs:
                for file in os.listdir(os.path.join(root, dir)):
                    if file.endswith('.json'):
                        yield file

    def create_pdf_folder(self, dir: str) -> None:
        print(f'Initializing {self.create_pdf_folder.__name__}')
        if not os.path.exists(dir):
            os.makedirs(dir)

    def create_png_folder(self, generator):
        print(f'Initializing {self.create_png_folder.__name__}')
        # stripping last element from path
        for elem in generator:
            dir_without_last = elem.split('/')[:-1]
            dir_without_last = '/'.join(dir_without_last)
            
            # creating new directory with _png suffix
            new_dir = dir_without_last + '_png'
        
        yield new_dir

    def convert_pdf_to_png(self, pdf_path, year):
        all_pdf_files: int = self.count_all_pdf_files_per_year()
        converted_files_counter: int = 0
        next_progress: int = 10

        try:
            pdf = pdfium.PdfDocument(pdf_path)
            n_pages = len(pdf)

            folder_path = pdf_path.rsplit('.', 1)[0] + '_png'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            for page_number in range(n_pages):
                page = pdf.get_page(page_number)
                pil_image = page.render(scale=5, rotation=0).to_pil()
                pil_image.save(f"{folder_path}/page_{page_number}.png")
            converted_files_counter += 1
            my_progress = (converted_files_counter / all_pdf_files) * 100

            if my_progress >= next_progress:
                print(f"{int(next_progress)}% of documents converted, year {year}.")
                next_progress += 10
            
        except pdfium.PdfiumError as e:
            print(f"Failed to load PDF document {pdf_path}: {e}")
        except Exception as e:
            print(f"An error occurred while converting {pdf_path} to PNG: {e}")

    def yield_json_files(self, year):
        print(f'Initializing {self.yield_json_files.__name__}')
        json_path = os.path.join(self.base_json_path, str(year))
        for root, _, files in os.walk(json_path):
            for file in files:
                if file.endswith('.json'):
                    yield os.path.join(root, file)

    def png_paths_creator(self, year) -> list[str]:
        print(f'Initializing {self.png_paths_creator.__name__}')
        pdf_path = os.path.join(self.base_pdf_path, str(year))
        pngs_0_list = []
        for root, _, files in os.walk(pdf_path):
            for file in files:
                if file.endswith('_0.png'):
                    pngs_0_list.append(os.path.join(root, file))
        return pngs_0_list

    def list_of_json_paths(self) -> list[str]:
        print(f'Initializing {self.list_of_json_paths.__name__}')
        my_list = []

        for root, dirs, files in os.walk(self.json_path):
            for dir in dirs:
                if dir == "2014":
                    for file in os.listdir(os.path.join(root, dir)):
                        if file.endswith('.json'):
                            my_list.append(os.path.join(root, dir, file))

        return my_list

    def json_text_debugger(self, iterator: iter, my_data: classmethod) -> None:
        print(f"Starting debugging...")

        for elem in iterator:
            json_text = my_data.clean_text_from_json(my_data.get_text_from_json(my_data.read_json_data(elem)))

            if len(json_text) != 0:
                print(f"{elem}")
                print(f"Json text first 100 characters: {json_text[:100]} \n")

        print(f"Debugging ended!")

    def pngs_list_debugger(self, my_list: list[str], my_data: classmethod) -> None:
        max_lcs = {}

        for elem in my_list[:1]:
            text = my_data.combine_text_to_one_string(my_data.clean_text(my_data.get_text_from_png(elem)))
            print(f"{elem} text: {text[:100]} \n")

            for file_path in elem:
                # file_path -> str
                max_lcs[f"{file_path}"] = self.longest_common_subsequence_dynamic(file_path, text)

            # max value
            max_value = max(max_lcs.values())
            print(f"Max lcs value {max_value}")

    def find_matching_dates(self, excel_1: str = 'extracted_dates.xlsx', excel_2: str = 'extracted_json_dates.xlsx', output_excel: str = 'matching_dates.xlsx') -> None:
        # Wczytaj pliki Excel do DataFrame'ów
        df1 = pd.read_excel(excel_1)
        df2 = pd.read_excel(excel_2)

        # Konwersja kolumny dat na format daty, z usunięciem wierszy bez daty
        df1['Extracted Date'] = pd.to_datetime(df1['Extracted Date'], errors='coerce').dt.date
        df2['Extracted Date'] = pd.to_datetime(df2['Extracted Date'], errors='coerce').dt.date

        df1 = df1.dropna(subset=['Extracted Date'])
        df2 = df2.dropna(subset=['Extracted Date'])

        # Znajdź dopasowane daty między dwoma DataFrame'ami
        matching_dates = pd.merge(df1, df2, on='Extracted Date', how='inner', suffixes=('_pdf', '_json'))

        # Zapisz wyniki do nowego pliku Excel
        matching_dates.to_excel(output_excel, index=False, columns=['Image folder path', 'JSON file path', 'Extracted Date', 'Text_json', 'Text'])
        print(f"Matching dates saved to {output_excel}")

    def calculate_cosine_similarity(self, excel_path = "matching_dates.xlsx", lemmatize = True):
        df = pd.read_excel(excel_path)

        nlp = spacy.load("pl_core_news_lg")

        def calculate_similarity(text1, text2):
            if lemmatize:
                doc1 = nlp(" ".join([token.lemma_ for token in nlp(text1)]))
                doc2 = nlp(" ".join([token.lemma_ for token in nlp(text2)]))
            else:
                doc1 = nlp(text1)
                doc2 = nlp(text2)
            return doc1.similarity(doc2)
        
        similarities = []
        for _, row in df.iterrows():
            similarity = calculate_similarity(row['Text'].lower(), row['Text_json'].lower())
            similarities.append(similarity)

        df['Cosine Similarity'] = similarities

        # update the excel file
        df.to_excel(excel_path, index=False, engine="openpyxl")

    def check_similarities(self, excel_path = "matching_dates.xlsx"):
        df = pd.read_excel(excel_path)

        for _, row in df.iterrows():
            if row['Cosine Similarity'] > 0.98:
                # wyświetlanie numeru wiersza
                print(f"Row number: {row.name}")
                print(f"PDF text: {row['Text'][:300]}")
                print(f"JSON text: {row['Text_json'][:300]}")
                print(f"Cosine similarity: {row['Cosine Similarity']}")

    def spacy_tester(self):
        nlp = spacy.load("pl_core_news_lg")
        df = pd.read_excel("matching_dates.xlsx")

        # get the first row, Text column
        text = df.iloc[0]['Text'].lower()
        print(f"Before lemmatization: {text}")

        doc = nlp(text)
        lemmatized_text = " ".join([token.lemma_ for token in doc])
        print(f"After lemmatization: {lemmatized_text}")
        doc2 = nlp(lemmatized_text)

        # compare the similarity between the original and lemmatized text
        print(f"Similarity between original and lemmatized text: {doc.similarity(doc2)}")

    def create_list_of_similarities(self) -> list[dict]:
        path: str = "./matching_dates.xlsx"

        if os.path.exists(path):
            df = pd.read_excel(path)
        else:
            print(f"File {path} does not exist.")
            return
        
        similarity_threshold: float = 0.98
        similarities: list = []

        # dict = {cosine_similarity: (image_folder_path, json_file_path)}
        for _, row in df.iterrows():
            if row['Cosine Similarity'] > similarity_threshold:
                my_dict: dict = {}
                my_dict[row['Cosine Similarity']] = (row['Image folder path'], row['JSON file path'])
                similarities.append(my_dict)

        # debug
        # print(f"Number of similar texts: {len(similarities)}")
        # print(f"Similarities: {similarities}")

        return similarities

    def find_start_end_each_page(self) -> pd.ExcelFile:
        my_list: list[dict] = self.create_list_of_similarities()

        # only first element for now
        first_elem = my_list[0]
        # print(f"First element: {first_elem}")

        # load first page from first element from first value
        png_folder_path = first_elem[list(first_elem.keys())[0]][0]
        # print(f"PNG folder path: {png_folder_path}")

        first_page_text = ""
        for root, dirs, files in os.walk(png_folder_path):
            for file in files:
                if file.endswith('.png'):
                    # get first page
                    first_page = os.path.join(root, file)
                    
                    # text from first page
                    first_page_text = self.get_text_from_png(first_page)
                    first_page_text = self.clean_text(first_page_text)
                    first_page_text = self.combine_text_to_one_string(first_page_text)
                    break

        print(f"First page text: {first_page_text}")

    @staticmethod
    def check_length_of_simple_file(image_folder_path: str = None) -> bool:
        counter: int = 0

        if not image_folder_path:
            raise ValueError("Path do not exist")

        for root, dirs, files in os.walk(image_folder_path):
            for elem in files:
                counter += 1

        if counter <= 5:
            return True
        else:
            return False
        
    @staticmethod
    def delete_past_jsons() -> None:
        root_dir: str = "JSON_files"

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                file_path: str = os.path.join(root, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error occured: {e} in function: delete_past_jsons")