from data import Data
from utils import Utils

def main():
    data = Data(json_path='lemkin-json-from-html', pdf_path='lemkin-pdf')
    utils = Utils(json_path='lemkin-json-from-html')

    # how many files are there in both directories
    print("Detected {} .json and {} .pdf files".format(data.number_of_files()[0], data.number_of_files()[1]))

    # yeilding files from json folder
    # print(utils.json_folder_iterator())

if __name__ == '__main__':
    main()
