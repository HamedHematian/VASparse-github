
import copy
import inspect
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import time

from transformers.cache_utils import Cache, DynamicCache
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from transformers.utils import ExplicitEnum, ModelOutput, is_accelerate_available, logging
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.generation.streamers import BaseStreamer

import math
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM,LlamaPreTrainedModel,Cache,DynamicCache

import warnings
from transformers.modeling_outputs import BaseModelOutputWithPast,CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from minigpt4.sparse.utils import batch_index_select, cluster_and_merge, attn_postprocess_rank, batch_attn_postprocess_rank

from transformers import AutoConfig

global sparse_count

def llamamodel_patching(model, model_config, pruning_loc=[2, 6, 15, 19]):
    setattr(model.llama_model.model, 'forward', new_forward.__get__(model.llama_model.model))
    base_model = model.llama_model.model
    base_model.pruning_loc = pruning_loc
    try:
        model_config_dict = AutoConfig.from_pretrained(model_config.merged_ckpt)
    except:
        model_config_dict = AutoConfig.from_pretrained(model_config.llama_model)

    base_model.num_layers = model_config_dict.num_hidden_layers
    base_model.num_forward = 0
    base_model.num_token_pool = 0

    base_model.init_token_total_shape = 664
    base_model.generate_process_count = 0

    base_model.t_token_idx = None

    decode_layers = model.llama_model.model.layers
    model.llama_model.pre_prompt_length_list = []

def llamamodel_patching_mplug_owl2(model, model_config, pruning_loc=[2, 6, 15, 19]):
    setattr(model.model.model, 'forward', new_forward_mplug_owl2.__get__(model.model.model))
    base_model = model.model.model
    base_model.pruning_loc = pruning_loc
    try:
        model_config_dict = AutoConfig.from_pretrained(model_config.merged_ckpt)
    except:
        model_config_dict = AutoConfig.from_pretrained(model_config.llama_model)

    base_model.num_layers = model_config_dict.num_hidden_layers
    base_model.num_forward = 0
    base_model.num_token_pool = 0

    base_model.init_token_total_shape = 664
    base_model.generate_process_count = 0

    base_model.t_token_idx = None

    decode_layers = model.model.model.layers
    model.model.pre_prompt_length_list = []


def new_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    image_shape = 576,
    pre_prompt_length_list = [],
    scale = 13.5,
    bias = 0.0,
    max_sentence_lenght = 32,
    sparse_kv_cache_rate = 0.9,
    prompt_length_image_text = 623,
    findings_kwargs = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    try:
        max_sentence_lenght = findings_kwargs['max_sentence_lenght']
        sparse_kv_cache_rate = findings_kwargs['sparse_kv_cache_rate']
        prompt_length_image_text = findings_kwargs['prompt_length_image_text']
    except:
        print("!!!!!"*10,"using the default setting..","!!!!!"*10)

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds

            
    if self.gradient_checkpointing and self.training:
        if use_cache:
            use_cache = False

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None


    if findings_kwargs and findings_kwargs.get('vasparse_outputs_base_layer', False):
        if findings_kwargs['base_contrastive_layer']==0:
            all_hidden_states += (hidden_states,)
            next_cache = next_decoder_cache if use_cache else None
            if not return_dict:
                return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )


    num_token = []
    B, L, _ = hidden_states.shape
    idx_sprase_layer = 0
    out_pred_prob = None
    init_n = self.init_token_total_shape + self.generate_process_count
    prev_decision = torch.ones(B, init_n, 1, dtype=hidden_states.dtype, device=hidden_states.device)
    policy = torch.ones(B, init_n, 1, dtype=hidden_states.dtype, device=hidden_states.device)

    v_token_start = pre_prompt_length_list[0] if len(pre_prompt_length_list) != 0 else 0
    text_token_start = v_token_start + image_shape
    v_token_num = image_shape

    v_t = hidden_states[:, v_token_start: text_token_start, :]
    t_t = hidden_states[:, text_token_start: , :]
    m_v_t = v_t @ t_t.transpose(1, 2) # [1, 576, 53]
    m_v_t = m_v_t.softmax(2).mean(1) # [1, 53]
    t_token_idx = torch.where(m_v_t > m_v_t.mean())
    if not self.t_token_idx:
        self.t_token_idx = t_token_idx

    layer_outputs= None
    if findings_kwargs and findings_kwargs.get('SparsePrefilling', False):
        pruning_loc = self.pruning_loc
    else:
        pruning_loc = []
    global sparse_count
    
    if past_key_values is None:
        print("*****"*10, f"max_sentence_lenght = {max_sentence_lenght}", "*****"*10)
        print('*****'*10, f'sparse_kv_cache_rate = {sparse_kv_cache_rate}', '*****'*10)
    

    input_attention_mask = attention_mask
    input_position_ids = position_ids
    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if findings_kwargs and findings_kwargs.get('vasparse_outputs_base_layer', False):
            if idx >= findings_kwargs['base_contrastive_layer']:
                break
        if attention_mask !=None and hidden_states.shape[-2] != attention_mask.shape[-1]:
            q_len = hidden_states.shape[-2]
            if q_len!=1:
                attention_mask = input_attention_mask[:, :, :hidden_states.shape[-2], :hidden_states.shape[-2]]
            else:
                attention_mask = None

        if idx in pruning_loc and len(pre_prompt_length_list) != 0 and hidden_states.shape[1] !=1:
            assert B==1, "vasparse only support Batch == 1 in Prefilling"
            assert output_attentions == True
            layer_outputs = decoder_layer(
                    hidden_states = hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
            )

            attn_logits = layer_outputs[1].clone().detach() 

            pred_score_vis, s_flag, relation_vis_text = attn_postprocess_rank(attn_logits, v_token_start, v_token_num, \
                text_token_start, t_token_idx, scale=scale, bias=bias) # B, L_v



            policy = torch.ones(B, hidden_states.shape[1], dtype=hidden_states.dtype, device=hidden_states.device)
            policy[:, v_token_start:text_token_start] = pred_score_vis.type(dtype = hidden_states.dtype)
            for batch in range(len(pre_prompt_length_list)):
                prompt_length = pre_prompt_length_list[batch]
                policy[batch,:prompt_length,] = 1
                text_token_start = prompt_length + image_shape
                policy[batch, text_token_start:,] = 1


            if s_flag:
                total_sparse_token_idx = torch.where(policy == 0)[1].unsqueeze(0)  
                total_sparse_token = batch_index_select(layer_outputs[0], total_sparse_token_idx) 
                
                merge_token_idx_stage1 = torch.where(pred_score_vis==0)[1]
                merge_token_stage1 = relation_vis_text[0][merge_token_idx_stage1]
                merge_token_num_stage1 = int(merge_token_idx_stage1.shape[0] * 0.3 ) + 1 # Top 30%
                merge_token_stage2_idx = merge_token_stage1.topk(merge_token_num_stage1)[1]
                
                merge_token_stage2 = total_sparse_token[:,merge_token_stage2_idx,:]
                if (cluster_num == 0) :
                    cluster_num = merge_token_stage2.shape[1]
                
                merge_sparse_token = cluster_and_merge(merge_token_stage2, cluster_num)  

                select_token_idx = torch.where(policy == 1)[1].unsqueeze(0)  # B, L_new
                select_token = batch_index_select(layer_outputs[0], select_token_idx)
                select_vis_token_num = pred_score_vis.sum()
                select_and_merge_token = torch.cat((select_token[:,:v_token_start+select_vis_token_num,:] ,
                        merge_sparse_token,
                        select_token[:,v_token_start+select_vis_token_num:,:])
                        ,dim=1
                )

                if output_attentions:
                    layer_outputs = (select_and_merge_token, layer_outputs[1], layer_outputs[2])  # B, L, C
                else:
                    layer_outputs = (select_and_merge_token, layer_outputs[1])  # B, L, C
                position_ids = input_position_ids[:, :len(select_token_idx[0])+cluster_num]
                prev_decision = policy
                v_token_num = pred_score_vis.sum() + cluster_num # B == 1
                text_token_start = v_token_start + v_token_num
            else:
                select_token_idx = torch.where(policy == 1)[1].unsqueeze(0)  # B, L_new
                if output_attentions:
                    layer_outputs = (batch_index_select(layer_outputs[0], select_token_idx), layer_outputs[1], layer_outputs[2])  # B, L, C
                else:
                    layer_outputs = (batch_index_select(layer_outputs[0], select_token_idx), layer_outputs[1])  # B, L, C
                position_ids = input_position_ids[:, :len(select_token_idx[0])]
                prev_decision = policy
                
                v_token_num = pred_score_vis.sum() # B == 1
                text_token_start = v_token_start + v_token_num

            idx_sprase_layer = idx_sprase_layer + 1
        else:

            if past_key_values is not None and prompt_length_image_text is not None:
                new_token_length_generated = past_key_values[idx][0].shape[-2] - prompt_length_image_text
                sentence_num = (new_token_length_generated ) // max_sentence_lenght
                if sentence_num >sparse_count and new_token_length_generated % max_sentence_lenght ==0:
                    sparse_count +=1
                    retain_length = int( (1- sparse_kv_cache_rate) * (sentence_num*max_sentence_lenght)) + prompt_length_image_text
                    merge_token_key = torch.mean(past_key_values[idx][0][:,:,prompt_length_image_text:retain_length,:], dim=2, keepdim=True)
                    merge_token_value = torch.mean(past_key_values[idx][1][:,:,prompt_length_image_text:retain_length,:], dim=2, keepdim=True)

                    past_key = torch.cat((past_key_values[idx][0][:,:,:prompt_length_image_text,:],merge_token_key, past_key_values[idx][0][:,:,retain_length:,:]), dim=2)
                    past_value = torch.cat((past_key_values[idx][1][:,:,:prompt_length_image_text,:], merge_token_value, past_key_values[idx][1][:,:,retain_length:,:]), dim=2)
                    past_key_value = (past_key, past_value)
                else:
                    past_key_value = past_key_values[idx] if past_key_values is not None else None
            else:
                sparse_count = 0
                past_key_value = past_key_values[idx] if past_key_values is not None else None
            if attention_mask==None:
                assert position_ids.shape[0] == 1
                position_ids = torch.tensor([[past_key_value[0].shape[-2]]], dtype=torch.int64).cuda()

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )


        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def new_forward_mplug_owl2(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    image_shape = 576,
    pre_prompt_length_list = [],
    scale = 13.5,
    bias = 0.0,
    max_sentence_lenght = 32,
    sparse_kv_cache_rate = 0.9,
    prompt_length_image_text = 623,
    findings_kwargs = None,
    modality_indicators: torch.Tensor = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    try:
        max_sentence_lenght = findings_kwargs['max_sentence_lenght']
        sparse_kv_cache_rate = findings_kwargs['sparse_kv_cache_rate']
        prompt_length_image_text = findings_kwargs['prompt_length_image_text']
    except:
        print("!!!!!"*10,"using the default setting..","!!!!!"*10)

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds

            
    if self.gradient_checkpointing and self.training:
        if use_cache:
            use_cache = False

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None


    if findings_kwargs and findings_kwargs.get('vasparse_outputs_base_layer', False):
        if findings_kwargs['base_contrastive_layer']==0:
            all_hidden_states += (hidden_states,)
            next_cache = next_decoder_cache if use_cache else None
            if not return_dict:
                return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )


    num_token = []
    B, L, _ = hidden_states.shape
    idx_sprase_layer = 0
    out_pred_prob = None
    init_n = self.init_token_total_shape + self.generate_process_count
    prev_decision = torch.ones(B, init_n, 1, dtype=hidden_states.dtype, device=hidden_states.device)
    policy = torch.ones(B, init_n, 1, dtype=hidden_states.dtype, device=hidden_states.device)

    v_token_start = pre_prompt_length_list[0] if len(pre_prompt_length_list) != 0 else 0
    text_token_start = v_token_start + image_shape
    v_token_num = image_shape

    v_t = hidden_states[:, v_token_start: text_token_start, :]
    t_t = hidden_states[:, text_token_start: , :]
    m_v_t = v_t @ t_t.transpose(1, 2) # [1, 576, 53]
    m_v_t = m_v_t.softmax(2).mean(1) # [1, 53]
    t_token_idx = torch.where(m_v_t > m_v_t.mean())
    if not self.t_token_idx:
        self.t_token_idx = t_token_idx

    layer_outputs= None
    if findings_kwargs and findings_kwargs.get('SparsePrefilling', False):
        pruning_loc = self.pruning_loc
    else:
        pruning_loc = []
    global sparse_count
    
    if past_key_values is None:
        print("*****"*10, f"max_sentence_lenght = {max_sentence_lenght}", "*****"*10)
        print('*****'*10, f'sparse_kv_cache_rate = {sparse_kv_cache_rate}', '*****'*10)
    

    input_attention_mask = attention_mask
    input_position_ids = position_ids
    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if findings_kwargs and findings_kwargs.get('vasparse_outputs_base_layer', False):
            if idx >= findings_kwargs['base_contrastive_layer']:
                break
        if attention_mask !=None and hidden_states.shape[-2] != attention_mask.shape[-1]:
            q_len = hidden_states.shape[-2]
            if q_len!=1:
                attention_mask = input_attention_mask[:, :, :hidden_states.shape[-2], :hidden_states.shape[-2]]
            else:
                attention_mask = None

        if idx in pruning_loc and len(pre_prompt_length_list) != 0 and hidden_states.shape[1] !=1:
            assert B==1, "vasparse only support Batch == 1 in Prefilling"
            assert output_attentions == True
            layer_outputs = decoder_layer(
                    hidden_states = hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
            )

            attn_logits = layer_outputs[1].clone().detach() 

            pred_score_vis, s_flag, relation_vis_text = attn_postprocess_rank(attn_logits, v_token_start, v_token_num, \
                text_token_start, t_token_idx, scale=scale, bias=bias) # B, L_v



            policy = torch.ones(B, hidden_states.shape[1], dtype=hidden_states.dtype, device=hidden_states.device)
            policy[:, v_token_start:text_token_start] = pred_score_vis.type(dtype = hidden_states.dtype)
            for batch in range(len(pre_prompt_length_list)):
                prompt_length = pre_prompt_length_list[batch]
                policy[batch,:prompt_length,] = 1
                text_token_start = prompt_length + image_shape
                policy[batch, text_token_start:,] = 1


            if s_flag:
                total_sparse_token_idx = torch.where(policy == 0)[1].unsqueeze(0)  
                total_sparse_token = batch_index_select(layer_outputs[0], total_sparse_token_idx) 
                
                merge_token_idx_stage1 = torch.where(pred_score_vis==0)[1]
                merge_token_stage1 = relation_vis_text[0][merge_token_idx_stage1]
                merge_token_num_stage1 = int(merge_token_idx_stage1.shape[0] * 0.3 ) + 1 # Top 30%
                merge_token_stage2_idx = merge_token_stage1.topk(merge_token_num_stage1)[1]
                
                merge_token_stage2 = total_sparse_token[:,merge_token_stage2_idx,:]
                if (cluster_num == 0) :
                    cluster_num = merge_token_stage2.shape[1]
                
                merge_sparse_token = cluster_and_merge(merge_token_stage2, cluster_num)  

                select_token_idx = torch.where(policy == 1)[1].unsqueeze(0)  # B, L_new
                select_token = batch_index_select(layer_outputs[0], select_token_idx)
                select_vis_token_num = pred_score_vis.sum()
                select_and_merge_token = torch.cat((select_token[:,:v_token_start+select_vis_token_num,:] ,
                        merge_sparse_token,
                        select_token[:,v_token_start+select_vis_token_num:,:])
                        ,dim=1
                )

                if output_attentions:
                    layer_outputs = (select_and_merge_token, layer_outputs[1], layer_outputs[2])  # B, L, C
                else:
                    layer_outputs = (select_and_merge_token, layer_outputs[1])  # B, L, C
                position_ids = input_position_ids[:, :len(select_token_idx[0])+cluster_num]
                prev_decision = policy
                v_token_num = pred_score_vis.sum() + cluster_num # B == 1
                text_token_start = v_token_start + v_token_num
            else:
                select_token_idx = torch.where(policy == 1)[1].unsqueeze(0)  # B, L_new
                if output_attentions:
                    layer_outputs = (batch_index_select(layer_outputs[0], select_token_idx), layer_outputs[1], layer_outputs[2])  # B, L, C
                else:
                    layer_outputs = (batch_index_select(layer_outputs[0], select_token_idx), layer_outputs[1])  # B, L, C
                position_ids = input_position_ids[:, :len(select_token_idx[0])]
                prev_decision = policy
                
                v_token_num = pred_score_vis.sum() # B == 1
                text_token_start = v_token_start + v_token_num

            idx_sprase_layer = idx_sprase_layer + 1
        else:

            if past_key_values is not None and prompt_length_image_text is not None:
                new_token_length_generated = past_key_values[idx][0].shape[-2] - prompt_length_image_text
                sentence_num = (new_token_length_generated ) // max_sentence_lenght
                if sentence_num >sparse_count and new_token_length_generated % max_sentence_lenght ==0:
                    sparse_count +=1
                    retain_length = int( (1- sparse_kv_cache_rate) * (sentence_num*max_sentence_lenght)) + prompt_length_image_text
                    merge_token_key = torch.mean(past_key_values[idx][0][:,:,prompt_length_image_text:retain_length,:], dim=2, keepdim=True)
                    merge_token_value = torch.mean(past_key_values[idx][1][:,:,prompt_length_image_text:retain_length,:], dim=2, keepdim=True)

                    past_key = torch.cat((past_key_values[idx][0][:,:,:prompt_length_image_text,:],merge_token_key, past_key_values[idx][0][:,:,retain_length:,:]), dim=2)
                    past_value = torch.cat((past_key_values[idx][1][:,:,:prompt_length_image_text,:], merge_token_value, past_key_values[idx][1][:,:,retain_length:,:]), dim=2)
                    past_key_value = (past_key, past_value)
                else:
                    past_key_value = past_key_values[idx] if past_key_values is not None else None
            else:
                sparse_count = 0
                past_key_value = past_key_values[idx] if past_key_values is not None else None
            if attention_mask==None:
                assert position_ids.shape[0] == 1
                position_ids = torch.tensor([[past_key_value[0].shape[-2]]], dtype=torch.int64).cuda()

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    modality_indicators,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    modality_indicators=modality_indicators,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )


        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )





def new_forward_single(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    image_shape = 576,
    pre_prompt_length_list = [],
    scale = 13.5,
    bias = 0.0,
    findings_kwargs = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )
    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            use_cache = False

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None
    num_token = []
    B, L, _ = hidden_states.shape
    idx_sprase_layer = 0
    out_pred_prob = None
    init_n = self.init_token_total_shape + self.generate_process_count
    prev_decision = torch.ones(B, init_n, 1, dtype=hidden_states.dtype, device=hidden_states.device)
    policy = torch.ones(B, init_n, 1, dtype=hidden_states.dtype, device=hidden_states.device)

    v_token_start = pre_prompt_length_list[0] if len(pre_prompt_length_list) != 0 else 0
    text_token_start = v_token_start + image_shape
    v_token_num = image_shape

    v_t = hidden_states[:, v_token_start: text_token_start, :]
    t_t = hidden_states[:, text_token_start: , :]
    m_v_t = v_t @ t_t.transpose(1, 2) # [1, 576, 53]
    m_v_t = m_v_t.softmax(2).mean(1) # [1, 53]
    t_token_idx = torch.where(m_v_t > m_v_t.mean())
    if not self.t_token_idx:
        self.t_token_idx = t_token_idx

    layer_outputs= None
    if findings_kwargs and findings_kwargs.get('SparsePrefilling', False):
        print("\033[41m!!!!! vasparse Pruning !!!!!!\033[0m")
        pruning_loc = self.pruning_loc
    else:
        pruning_loc = []

    input_attention_mask = attention_mask
    input_position_ids = position_ids



    for idx, decoder_layer in enumerate(self.layers):
        if attention_mask !=None and hidden_states.shape[-2] != attention_mask.shape[-1]:
            q_len = hidden_states.shape[-2]
            if q_len!=1:
                attention_mask = input_attention_mask[:, :, :hidden_states.shape[-2], :hidden_states.shape[-2]]
            else:
                attention_mask = None

        if idx in pruning_loc and len(pre_prompt_length_list) != 0 and hidden_states.shape[1] !=1:
            assert output_attentions == True
            layer_outputs = decoder_layer(
                    hidden_states = hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
            )

            attn_logits = layer_outputs[1].clone().detach() # 实际上就是注意力weight, output_attentions=True 


            pred_score_vis, s_flag, relation_vis_text = batch_attn_postprocess_rank(attn_logits, v_token_start, v_token_num, \
                text_token_start, t_token_idx, scale=scale, bias=bias) # B, L_v


            policy = torch.ones(B, hidden_states.shape[1], dtype=hidden_states.dtype, device=hidden_states.device)
            policy[:, v_token_start:text_token_start] = pred_score_vis.type(dtype = hidden_states.dtype)
            for batch in range(len(pre_prompt_length_list)):
                prompt_length = pre_prompt_length_list[batch]
                policy[batch,:prompt_length,] = 1
                text_token_start = prompt_length + image_shape
                policy[batch, text_token_start:,] = 1

            if s_flag:

                total_sparse_token_idx = torch.where(policy == 0)[1].unsqueeze(0)  
                total_sparse_token = batch_index_select(layer_outputs[0], total_sparse_token_idx) 
                
                merge_token_idx_stage1 = torch.where(pred_score_vis==0)[1]
                merge_token_stage1 = relation_vis_text[0][merge_token_idx_stage1]
                merge_token_num_stage1 = int(merge_token_idx_stage1.shape[0] * 0.3 ) + 1 # Top 30%
                merge_token_stage2_idx = merge_token_stage1.topk(merge_token_num_stage1)[1]
                
                merge_token_stage2 = total_sparse_token[:,merge_token_stage2_idx,:]
                if (cluster_num == 0) :
                    cluster_num = merge_token_stage2.shape[1]
                
                merge_sparse_token = cluster_and_merge(merge_token_stage2, cluster_num)  

                select_token_idx = torch.where(policy == 1)[1].unsqueeze(0)  # B, L_new
                select_token = batch_index_select(layer_outputs[0], select_token_idx)
                select_vis_token_num = pred_score_vis.sum()
                select_and_merge_token = torch.cat((select_token[:,:v_token_start+select_vis_token_num,:] ,
                        merge_sparse_token,
                        select_token[:,v_token_start+select_vis_token_num:,:])
                        ,dim=1
                )

                if output_attentions:
                    layer_outputs = (select_and_merge_token, layer_outputs[1], layer_outputs[2])  # B, L, C
                else:
                    layer_outputs = (select_and_merge_token, layer_outputs[1])  # B, L, C
                position_ids = input_position_ids[:, :len(select_token_idx[0])+cluster_num]
                prev_decision = policy
                v_token_num = pred_score_vis.sum() + cluster_num # B == 1
                text_token_start = v_token_start + v_token_num
            else:
                select_token_idx = torch.where(policy == 1)[1].unsqueeze(0)  # B, L_new
                if output_attentions:
                    layer_outputs = (batch_index_select(layer_outputs[0], select_token_idx), layer_outputs[1], layer_outputs[2])  # B, L, C
                else:
                    layer_outputs = (batch_index_select(layer_outputs[0], select_token_idx), layer_outputs[1])  # B, L, C
                position_ids = input_position_ids[:, :len(select_token_idx[0])]
                prev_decision = policy
                
                v_token_num = pred_score_vis.sum() # B == 1
                text_token_start = v_token_start + v_token_num

            idx_sprase_layer = idx_sprase_layer + 1
        else:

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if attention_mask==None:

                assert position_ids.shape[0] == 1
                position_ids = torch.tensor([[past_key_value[0].shape[-2]]], dtype=torch.int64).cuda()

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )


        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
