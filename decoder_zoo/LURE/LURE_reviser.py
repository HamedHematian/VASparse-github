import argparse
import os
import random
import sys
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION, Conversation, SeparatorStyle

from PIL import Image
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="/eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--input_caption", help="path to input caption.")
    parser.add_argument("--input_image", help="path to image file.")
    parser.add_argument("--output_file", help="path to output file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))


results = []
input_dir =  args.input_image
output_file = args.output_file 
input_caption = args.input_caption
caption_data = []
with open(input_caption, 'r', encoding='utf-8') as f:
    for line in f:
        caption_data.append(json.loads(line.strip()))


prefix = "COCO_val2014_"
max_new_tokens = 64
# qs = "Please describe this image in detail."

with torch.no_grad():
    with open(output_file, "a+") as f:
        # for filename in tqdm(os.listdir(input_dir)):
        for idx, pair in tqdm(enumerate(caption_data), total=len(caption_data)):
            img_id = pair['image_id']
            caption = pair['caption']
            image_id = img_id
            img_id = str(img_id).zfill(12)
            image_path = input_dir + prefix + img_id + '.jpg'
            this_question = caption
            # print(this_question)
            chat_state = Conversation(
                system='Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.', 
                roles=('Human', 'Assistant'), 
                messages=[['Human', '<Img><ImageHere></Img> ' + 'According to the picture, remove the information that does not exist in the following description: ' + this_question]], 
                offset=2, 
                sep_style=SeparatorStyle.SINGLE, 
                sep='###', 
                sep2=None, 
                skip_next=False, 
                conv_id=None
            )
            # image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path).convert('RGB')
            img_list = []
            image = chat.vis_processor(image).unsqueeze(0).to('cuda:{}'.format(args.gpu_id))
            image_emb, _, = chat.model.encode_img(image)
            img_list.append(image_emb)
            output = chat.answer(chat_state, img_list, max_new_tokens)

            # float_list = [tensor.item() for tensor in plist]
            result = {"image_id": image_id, "question": this_question, "caption": output, "model": "LURE"}
            print(result)
            json.dump(result, f)
            f.write('\n')
            f.flush()
f.close()

    
