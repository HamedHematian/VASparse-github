import random
import numpy as np
import torch
import json
from decoder_zoo.HALC.context_density.detector import Detector
from transformers import Owlv2Processor, Owlv2ForObjectDetection, AutoTokenizer, AutoModelForCausalLM
from types import SimpleNamespace
from PIL import Image, ImageDraw
import spacy
from torch.nn import functional as F
from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipModel
import random
from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from PIL import Image, ImageFilter
import hpsv2
from mplug_owl2.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)



exempt_word_list = ["image", "side", "background", "feature", "features", "center", 
                    "left", "right", "scene", "view", "s", "Birthday", "detail", "red",
                    "white", "cat", "horse", "bus", "group", "manner", "her", "birds", 
                    "teddy", "stack", "cell", "toaster", "mirror", "captures"]

add_word_list = ["sink", "microwave", "toaster", "puppy", "bottle", "table", "oven", 
                "orange", "toothbrush", "cars"]



tagging = spacy.load("en_core_web_sm")

entity=  "surfboard"
doc = tagging(entity)
print(doc)