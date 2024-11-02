# Przetwarzanie skanów ustaw prawnych do formy ustrukturyzowanej
# Turbo ważne notatki do wirual venva
- przy próbie instalacji AutoAWQ automatycznie instaluje się torch 2.3.1 (ale bez torchvision i torchaudio)
- najpierw instalować torch 2.3.1, potem flash attention (wraz z packaging oraz ninja) i na koniec AutoAWQ

## To DO:
- ~~łączyć w jedną całość wiele stron, nie tylko tekst z pierwszej z nich~~
- ~~lepszy pre processing tekstu~~
- ~~zmienić ściezke pamieci .cache na SCRATCHA~~
- ~~ulepszyc sposob i miejsce zapisu jsonów -> dodac date do nazwy i nowy folder~~
- ~~zamiast tworzyć jeden `.json` przy użyciu `json.dump()` tworzyć wiele `.json` i zapisywać je w folderze, później łączyć przez `Qwen-2.5`~~
- podpiąć [dSpy](https://github.com/stanfordnlp/dspy)
- ~~podpiąć `Qwen 2.5` do automatycznego parsowania jsonów ze sobą~~
- dodać autonaprawianie `.json`-ów w `Qwen 2.5`
- dodać przykładowego `.json-a` do wiadomości w funkcji `dataset`

## Struktura folderów

- lemkin-pdf
  |
  |
  - subfolder1
  |
  |
    - subfolder2
    |
    |
    - pdf
    - _png folder

## Do przeczytania i zrobienia w tym semestrze:
- [dSpy](https://github.com/stanfordnlp/dspy)
- [MLflow](https://mlflow.org/#core-concepts)
- [LoRa](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)
- [LoRa paper](https://arxiv.org/abs/2106.09685)
- [Adapters paper](https://arxiv.org/abs/2304.01933)
- [Adapters Huggigface](https://huggingface.co/docs/hub/adapters)

## ViT -> Vision Transformer
- [Pierwszy paper](https://paperswithcode.com/method/vision-transformer)
- [Objaśnienie na medium](https://medium.com/@hansahettiarachchi/unveiling-vision-transformers-revolutionizing-computer-vision-beyond-convolution-c410110ef061)
- [Hugging Face](https://huggingface.co/docs/transformers/model_doc/vit)
- [MGP-STR](https://huggingface.co/docs/transformers/model_doc/mgp-str#mgp-str)
- [Objaśnienie PaddleOCR](https://vinod-baste.medium.com/unlocking-the-power-of-paddleocr-4141544f8dba)
- [PaddleOCR python](https://pypi.org/project/paddleocr/)
- [parseq](https://github.com/baudm/parseq)
- [trocr](https://github.com/microsoft/unilm/tree/master/trocr)
- [microsoft trocr](https://huggingface.co/microsoft/trocr-base-handwritten)
- [layoutlm3](https://huggingface.co/docs/transformers/model_doc/layoutlmv3)
- [PaliGemma – Google's Cutting-Edge Open Vision Language Model](https://huggingface.co/blog/paligemma)
- [mychen76/invoice-and-receipts_donut_v1](https://huggingface.co/mychen76/invoice-and-receipts_donut_v1)

## Przykładowe papery -> na później
- [OCR z NLP corektą](https://ieeexplore-1ieee-1org-1000047nk00e3.wbg2.bg.agh.edu.pl/document/10463478)
- [Development of Extensive Polish Handwritten Characters Database for Text Recognition Research](http://www.astrj.com/Development-of-Extensive-Polish-Handwritten-Characters-Database-for-Text-Recognition,122567,0,2.html)
- [PHCD katedra informatyki Lublin](https://cs.pollub.pl/phcd/)
- [Tworzenie bazy danych dla polskiego OCR](http://www.astrj.com/pdf-122567-53925?filename=Development%20of%20Extensive.pdf)
- [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/pdf/1704.03155v2)
- [Text detection ](https://paperswithcode.com/task/text-detection)
- [Robust Scene Text Detection and Recognition: Introduction](https://developer.nvidia.com/blog/robust-scene-text-detection-and-recognition-introduction/)
- [Scene Text Detection](https://paperswithcode.com/task/scene-text-detection)
- [Vision Transformer for Fast and Efficient Scene Text Recognition](https://arxiv.org/pdf/2105.08582)
- [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://paperswithcode.com/paper/trocr-transformer-based-optical-character)
- [TextMonkey: An OCR-Free Large Multimodal Model for Understanding Document](https://arxiv.org/pdf/2403.04473)
- [CORD: A Consolidated Receipt Dataset for Post-OCR Parsing](https://openreview.net/pdf?id=SJl3z659UH)

## Biblioteki
- [paddleocr](https://pypi.org/project/paddleocr/)
- [tesseract na githubie](https://github.com/tesseract-ocr/tesseract)
- [tesseractocr na pypi](https://github.com/sirfz/tesserocr)
- [ocrd-tesserocr](https://pypi.org/project/ocrd-tesserocr/)
- [pytesseract](https://pypi.org/project/pytesseract/)
- [pylcs](https://pypi.org/project/pylcs/)
- [jsonformer](https://github.com/1rgs/jsonformer)

## Opisy modeli, np. na Medium
- [The Role of LayoutLMv3 in Document Layout Understanding in 2024](https://medium.com/ubiai-nlp/the-role-of-layoutlmv3-in-document-layout-understanding-in-2024-46d505105cfb)
- [Tesseract OCR: What Is It, and Why Would You Choose It in 2024?](https://www.klippa.com/en/blog/information/tesseract-ocr/)
- [OCR Unlocked: A Guide to Tesseract in Python with Pytesseract and OpenCV](https://nanonets.com/blog/ocr-with-tesseract/)
- [Convert data from PDFs to JSON outputs](https://medium.com/nanonets/convert-data-from-pdfs-to-json-outputs-4bf32d50cfd2)
- [Finetune LLM to convert a receipt image to json or xml](https://mychen76.medium.com/finetune-llm-to-convert-a-receipt-image-to-json-or-xml-3f9a6237e991)
- [PaliGemma Vision-Language Model](https://medium.com/@nimritakoul01/paligemma-vision-language-model-22693c2a4dec)
- [Finetune LLM to convert a receipt image to json or xml](https://mychen76.medium.com/finetune-llm-to-convert-a-receipt-image-to-json-or-xml-3f9a6237e991)
