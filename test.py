import os
import regex as re

def sorting_by_page_number(png_path: str = None) -> int:
    split1: str = png_path.split("/")[-1]
    split2 = split1.split(".")[0]
    split3 = split2.split("_")[-1]

    return int(split3)

def get_pngs_path_from_folder(folder_path: str = None) -> list[str]:
    folder_path: str = "lemkin-pdf/2014/WDU20140000596/O/D20140596_png"
    pngs_paths: list[str] = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):
                pngs_paths.append(os.path.join(root, file))

    return sorted(pngs_paths, key=sorting_by_page_number())

def get_dataset() -> list[list[dict]]:
    dataset: list[list[dict]] = []
    pngs_paths: list[str] = ["lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_0.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_1.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_2.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_3.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_4.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_5.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_6.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_7.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_8.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_9.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_10.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_11.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_12.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_13.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_14.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_15.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_16.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_17.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_18.png", "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_19.png"]
    # pngs_paths = get_pngs_path_from_folder()
    max_batch_threshold: int = 15  # You can adjust this number as needed

    for i in range(0, len(pngs_paths), max_batch_threshold):
        current_batch = pngs_paths[i:i+max_batch_threshold]
        current_message_content: list[dict] = []

        for current_image_path in current_batch:
            root_image_path: str = os.path.relpath(current_image_path)
            current_message_content.append({
                "type": "image",
                "image": root_image_path
            })

        # Add the text instruction at the end of each message content
        current_message_content.append({
            "type": "text",
            "text": "Make a one, hierarchical .json from the image. Combine it with other messages. Polish language only. Leave only json characters"
        })

        # Construct the message
        message = [
            {
                "role": "user",
                "content": current_message_content
            },
        ]

        dataset.append(message)

    return dataset

print(get_dataset())