import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-72B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    cache_dir = "/net/scratch/hscra/plgrid/plgkruczek/.cache",
)

# default processer
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# Messages containing multiple images and a text query
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_0.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_1.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_2.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_3.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_4.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_5.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_6.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_7.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_8.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_9.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_10.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_11.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_12.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_13.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_14.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_15.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_16.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_17.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_18.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_19.png"},
            {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_20.png"},
            # {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_21.png"},
            # {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_22.png"},
            # {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_23.png"},
            # {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_24.png"},
            # {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_25.png"},
            # {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_26.png"},
            # {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_27.png"},
            # {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_28.png"},
            # {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_29.png"},
            # {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_30.png"},
            # {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_31.png"},
            # {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_32.png"},
            # {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_33.png"},
            # {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_34.png"},
            # {"type": "image", "image": "lemkin-pdf/2014/WDU20140000596/O/D20140596_png/page_35.png"},
            {"type": "text", "text": "Make a one, hierarchical .json from all images."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)