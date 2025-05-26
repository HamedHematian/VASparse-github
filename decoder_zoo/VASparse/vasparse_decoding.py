
import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from torch.nn import functional as F
import numpy as np
import random
import time
import gc
from copy import deepcopy

from minigpt4.sparse.utils import batch_index_select, cluster_and_merge, attn_postprocess_rank, attn_postprocess_rank_vasparse, process_kv_cache, attn_postprocess_rank_vasparse_visual, process_kv_cache_shared


from transformers import (
    LogitsProcessorList,
    StoppingCriteriaList,
)

from transformers import(
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
)

from transformers import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
    BeamScorer,
    BeamSearchScorer,
)

from transformers.generation import (
    GreedySearchDecoderOnlyOutput,
    ContrastiveSearchEncoderDecoderOutput,
    SampleDecoderOnlyOutput,
    BeamSearchDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSampleDecoderOnlyOutput,
    GreedySearchEncoderDecoderOutput,
)
from transformers.utils import ModelOutput

GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]


class BaseStreamer:
    """
    Base class from which `.generate()` streamers should inherit.
    """

    def put(self, value):
        """Function that is called by `.generate()` to push new tokens"""
        raise NotImplementedError()

    def end(self):
        """Function that is called by `.generate()` to signal the end of generation"""
        raise NotImplementedError()

def validate_stopping_criteria(stopping_criteria: StoppingCriteriaList, max_length: int) -> StoppingCriteriaList:
    stopping_max_length = stopping_criteria.max_length
    new_stopping_criteria = deepcopy(stopping_criteria)
    if stopping_max_length is not None and stopping_max_length != max_length:
        warnings.warn("You set different `max_length` for stopping criteria and `max_length` parameter", UserWarning)
    elif stopping_max_length is None:
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    return new_stopping_criteria


def _extract_past_from_model_output(self, outputs: ModelOutput, standardize_cache_format: bool = False):
    past_key_values = None
    if "past_key_values" in outputs:
        past_key_values = outputs.past_key_values
    elif "mems" in outputs:
        past_key_values = outputs.mems
    elif "past_buckets_states" in outputs:
        past_key_values = outputs.past_buckets_states

    if standardize_cache_format and hasattr(self, "_convert_to_standard_cache"):
        batch_size = outputs.logits.shape[0]
        past_key_values = self._convert_to_standard_cache(past_key_values, batch_size=batch_size)
    return past_key_values

def _update_model_kwargs_for_generation(
    self,
    outputs: ModelOutput,
    model_kwargs: Dict[str, Any],
    is_encoder_decoder: bool = False,
    standardize_cache_format: bool = False,
) -> Dict[str, Any]:
    model_kwargs["past_key_values"] = self._extract_past_from_model_output(
        outputs, standardize_cache_format=standardize_cache_format
    )
    if getattr(outputs, "state", None) is not None:
        model_kwargs["state"] = outputs.state

    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    if not is_encoder_decoder:
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
    else:
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = torch.cat(
                [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                dim=-1,
            )

    return model_kwargs

def _update_model_kwargs_for_vasparse_contrastive_decoding(
    self,
    outputs: ModelOutput,
    new_past_key_values, 
    model_kwargs: Dict[str, Any],
    is_encoder_decoder: bool = False,
    standardize_cache_format: bool = False,
) -> Dict[str, Any]:
    model_kwargs["past_key_values"] = new_past_key_values

    if getattr(outputs, "state", None) is not None:
        model_kwargs["state"] = outputs.state

    if "token_type_ids" in model_kwargs:
        token_type_ids = model_kwargs["token_type_ids"]
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

    if not is_encoder_decoder:
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
    else:
        if "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = torch.cat(
                [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                dim=-1,
            )

    return model_kwargs

def process_batch_inputs(self, input_ids, model_kwargs, output_attentions, output_hidden_states, early_exit_layers, findings_kwargs):


    batch_size = input_ids.shape[0]

    all_dict_outputs = []
    all_outputs = []

    for batch_idx in range(batch_size):
        single_input_ids = input_ids[batch_idx].unsqueeze(0)
        single_model_kwargs = {k: (v[batch_idx].unsqueeze(0) if isinstance(v, torch.Tensor) and v.shape[0] == batch_size else v) 
                               for k, v in model_kwargs.items()}

        model_inputs = self.prepare_inputs_for_generation(single_input_ids, **single_model_kwargs)

        import ipdb
        ipdb.set_trace()

        dict_outputs, outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            early_exit_layers=early_exit_layers,
            findings_kwargs=findings_kwargs,
        )

        all_dict_outputs.append(dict_outputs)
        all_outputs.append(outputs)

    combined_dict_outputs = {k: torch.cat([d[k] for d in all_dict_outputs], dim=0) for k in all_dict_outputs[0]}
    combined_outputs = torch.cat(all_outputs, dim=0)

    return combined_dict_outputs, combined_outputs

from transformers.modeling_outputs import CausalLMOutputWithPast
import torch

def merge_outputs(all_outputs):

    all_loss = []
    all_logits = []
    all_past_key_values = []
    all_hidden_states = []
    all_attentions = []

    batch_size = len(all_outputs)  # 确定 batch 大小

    for outputs in all_outputs:
        if outputs.loss is not None:
            all_loss.append(outputs.loss.unsqueeze(0))
        
        all_logits.append(outputs.logits.unsqueeze(0))

        if outputs.past_key_values is not None:
            all_past_key_values.append(outputs.past_key_values)

        if outputs.hidden_states is not None:
            all_hidden_states.append(outputs.hidden_states)

        if outputs.attentions is not None:
            all_attentions.append(outputs.attentions)

    combined_loss = None
    if len(all_loss) > 0:
        combined_loss = torch.cat(all_loss, dim=0)  

    combined_logits = torch.cat(all_logits, dim=0)  

    combined_past_key_values = None
    if all_past_key_values:
        combined_past_key_values = []
        for layer_idx in range(len(all_past_key_values[0])):
            layer_past_key_values = (
                torch.cat([all_past_key_values[batch_idx][layer_idx][0] for batch_idx in range(batch_size)], dim=0),  # key
                torch.cat([all_past_key_values[batch_idx][layer_idx][1] for batch_idx in range(batch_size)], dim=0)   # value
            )
            combined_past_key_values.append(layer_past_key_values)

        combined_past_key_values = tuple(combined_past_key_values)


    combined_hidden_states = None
    if all_hidden_states:
        num_layers = len(all_hidden_states[0])
        combined_hidden_states = []
        for layer_idx in range(num_layers):
            combined_layer = torch.cat([all_hidden_states[batch_idx][layer_idx] for batch_idx in range(batch_size)], dim=0)
            combined_hidden_states.append(combined_layer)
        combined_hidden_states = tuple(combined_hidden_states)


    combined_attentions = None
    if all_attentions:
        num_layers = len(all_attentions[0])
        combined_attentions = []
        for layer_idx in range(num_layers):
            combined_layer = torch.cat([all_attentions[batch_idx][layer_idx] for batch_idx in range(batch_size)], dim=0)
            combined_attentions.append(combined_layer)
        combined_attentions = tuple(combined_attentions)

    combined_outputs = CausalLMOutputWithPast(
        loss=combined_loss,
        logits=combined_logits,
        past_key_values=combined_past_key_values,
        hidden_states=combined_hidden_states,
        attentions=combined_attentions,
    )

    return combined_outputs

def replace_kv_prefix(past_key_values_1, past_key_values_2, a_list):

    modified_past_key_values_1 = []

    for layer_idx, (kv1, kv2) in enumerate(zip(past_key_values_1, past_key_values_2)):
        a = a_list[layer_idx]

        key1, value1 = kv1
        key2, value2 = kv2

        key1 = key1.clone()
        value1 = value1.clone()

        new_key1 = torch.cat([key2, key1[:, :, a:, :]], dim=2)

        new_value1 = torch.cat([value2, value1[:, :, a:, :]], dim=2)

        modified_kv = (new_key1, new_value1)
        modified_past_key_values_1.append(modified_kv)

    return tuple(modified_past_key_values_1)


def replace_kv_prefix_shared(past_key_values_1, past_key_values_2, a_list):
    return past_key_values_1



def relative_top_filter(
    scores: torch.FloatTensor,
    relative_top: float = 0.1,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:


    scores_normalized = scores.log_softmax(dim=-1)
    sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep - 1]
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    scores_normalized[scores_normalized < probs_thresh] = filter_value
    return scores_normalized


def vasparse_search_contrastive_decoding(
    self,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    mature_layer: int,
    base_layer: Optional[int] = None,
    candidate_premature_layers: Optional[List[int]] = None,
    relative_top: float = 0.1,
    beam_size: Optional[Union[int, List[int]]] = None,
    findings_kwargs: Optional[dict] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: Optional[bool] = False,
    mask_rate = 0.5,
    contrastive_rate = 0.1,
    base_contrastive_layer=32, 
    base_layer_not_sure=  False,
    detector_used=  True,
    **model_kwargs,
) -> Union[BeamSearchOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.beam_search`] directly. Use generate()
    instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        beam_scorer (`BeamScorer`):
            An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
            sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.


    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForSeq2SeqLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     BeamSearchScorer,
    ... )
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

    >>> encoder_input_str = "translate English to German: How old are you?"
    >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


    >>> # lets run beam search using 3 beams
    >>> num_beams = 3
    >>> # define decoder start token ids
    >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
    >>> input_ids = input_ids * model.config.decoder_start_token_id

    >>> # add encoder_outputs to model keyword arguments
    >>> model_kwargs = {
    ...     "encoder_outputs": model.get_encoder()(
    ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
    ...     )
    ... }

    >>> # instantiate beam scorer
    >>> beam_scorer = BeamSearchScorer(
    ...     batch_size=1,
    ...     num_beams=num_beams,
    ...     device=model.device,
    ... )

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
    ...     ]
    ... )

    >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ['Wie alt bist du?']
    ```"""

    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    if len(stopping_criteria) == 0:
        warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only

    if base_layer is not None and candidate_premature_layers is None:
        early_exit_layers = [base_layer, mature_layer]
        num_base_layers = 1
        premature_layer_dist = {}
    elif candidate_premature_layers is not None:
        early_exit_layers = candidate_premature_layers + [mature_layer]
        num_base_layers = len(candidate_premature_layers)
        premature_layer_dist = {l: 0 for l in candidate_premature_layers}
    else:
        raise ValueError("You must specify either `base_layer` or `candidate_premature_layers`")

    info_dict = {}
    

    assert findings_kwargs is not None and findings_kwargs.get('vasparse_assistant_helper', False)
    vasparse_helper = findings_kwargs['vasparse_assistant_helper']
    beam_size = findings_kwargs['beam_size']
    beam_last_word_flag = [None] * beam_size
    beam_current_word = [None] * beam_size
    full_dict_outputs, full_outputs = None, None
    sparse_dict_outputs, sparse_outputs = None, None
    prefilling_prompt_len = 0
    prefilling_prompt_len_list = []
    start_time, end_time = None, None
    total_time = 0.0
    total_count_token = 0
    shared_policy = True # You can use non shared implementation for shared performance, but it will result in lower efficiency

    
    while True:
        if synced_gpus:
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            if this_peer_finished_flag.item() == 0.0:
                break

        batch_process_vasparse_size = input_ids.shape[0]
        if not model_kwargs.get('past_key_values', False) and not shared_policy:
            all_dict_outputs = []
            all_outputs = []

            for batch_idx in range(batch_process_vasparse_size):
                single_input_ids = input_ids[batch_idx].unsqueeze(0)
                single_model_kwargs = {k: (v[batch_idx].unsqueeze(0) if isinstance(v, torch.Tensor) and v.shape[0] == batch_process_vasparse_size else v) 
                                    for k, v in model_kwargs.items()}

                model_inputs = self.prepare_inputs_for_generation(single_input_ids, **single_model_kwargs)
        
                sparse_dict_outputs, sparse_outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    early_exit_layers=early_exit_layers,
                    findings_kwargs=findings_kwargs,
                )

                all_dict_outputs.append(sparse_dict_outputs)
                all_outputs.append(sparse_outputs)

            sparse_dict_outputs = {k: torch.cat([d[k] for d in all_dict_outputs], dim=0) for k in all_dict_outputs[0]}
            sparse_outputs = merge_outputs(all_outputs)
            prefilling_prompt_len_list = [k.shape[-2] for (k,v) in sparse_outputs.past_key_values]
            


            

        if not model_kwargs.get('past_key_values', False):
            findings_kwargs['SparsePrefilling'] = False

            if not full_outputs:
                all_dict_outputs = []
                all_outputs = []

                for batch_idx in range(batch_process_vasparse_size):
                    single_input_ids = input_ids[batch_idx].unsqueeze(0)
                    single_model_kwargs = {k: (v[batch_idx].unsqueeze(0) if isinstance(v, torch.Tensor) and v.shape[0] == batch_process_vasparse_size else v) 
                                        for k, v in model_kwargs.items()}

                    model_inputs = self.prepare_inputs_for_generation(single_input_ids, **single_model_kwargs)
            
                    full_dict_outputs, full_outputs = self(
                        **model_inputs,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        early_exit_layers=early_exit_layers,
                        findings_kwargs=findings_kwargs,
                    )

                    all_dict_outputs.append(full_dict_outputs)
                    all_outputs.append(full_outputs)

                full_dict_outputs = {k: torch.cat([d[k] for d in all_dict_outputs], dim=0) for k in all_dict_outputs[0]}
                full_outputs = merge_outputs(all_outputs)
                prefilling_prompt_len = full_outputs.attentions[0].shape[-2]

                outputs = full_outputs
                dict_outputs = full_dict_outputs
            else:
                assert False, "decoding errors"
                full_dict_outputs, full_outputs = self( # attentions (32, torch.Size([2, 32, 623, 623])) kv cache (32, 2, torch.Size([2, 32, 623, 128]) )
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    early_exit_layers=early_exit_layers,
                    findings_kwargs = findings_kwargs,
                )

            findings_kwargs['SparsePrefilling'] = True
        else:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            dict_outputs, outputs = self( 
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                early_exit_layers=early_exit_layers,
                findings_kwargs = findings_kwargs,
            )

        if not model_kwargs.get('past_key_values', False):
            if start_time is None:
                start_time = time.perf_counter() 
        total_count_token += 1 
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        if detector_used:
            beam_next_tokens_new = torch.argmax(next_token_scores, dim=-1)
            VH_is_detected = False
            sparse_ratios = []
            softmax_normalized_difference_list = []
            for bs in range(next_token_logits.shape[0]):
                beam_last_word_flag[bs] = vasparse_helper.check_word_complete(beam_next_tokens_new[:, None][bs][:, None])
                if beam_last_word_flag[bs]:
                    entity = vasparse_helper.get_last_word(beam_next_tokens_new[bs])
                    beam_current_word[bs] = entity

                    embeds_list, detect_info = vasparse_helper.visual_aware_region_embedding(entity)

                    if detect_info['status'] == 'not-detected':
                        sparse_ratios.append(1)
                        VH_is_detected = True
                    elif detect_info['status'] == 'activated' and embeds_list:
                        ori_image, mask_image = embeds_list[0], embeds_list[1]
                        ori_image_token, mask_image_token = self.encode_images(ori_image), self.encode_images(mask_image)
                        difference = 1- F.cosine_similarity(ori_image_token, mask_image_token, dim=-1).squeeze(0)  # shape: [1, 576]
                        softmax_normalized_difference = F.softmax(difference, dim=0)
                        softmax_normalized_difference_list.append(softmax_normalized_difference)
            
        if VH_is_detected: #VH_is_detected:
            policy_list = []
            aware_is = False
            if aware_is:

                for layer_i in range(len(full_outputs['attentions'])):

                    image_shape = 576
                    v_token_start = self.pre_prompt_length_list[0] if len(self.pre_prompt_length_list) != 0 else 0
                    text_token_start = v_token_start + image_shape
                    v_token_num = image_shape

                    attn_logits = full_outputs['attentions'][layer_i].clone().detach() # 实际上就是注意力weight, output_attentions=True torch.Size([2, 32, 623, 623])

                    pred_score_vis, s_flag, relation_vis_text = attn_postprocess_rank_vasparse_visual(attn_logits, v_token_start, v_token_num, \
                        text_token_start, self.model.t_token_idx, sparse_ratio=mask_rate, scale=13.5, bias=0.0) # B, L_v
                    
                    L_input = attn_logits.shape[-1]
                    B = attn_logits.shape[0]
                    policy = torch.ones(B, L_input, dtype=full_outputs['attentions'][layer_i].dtype, device=self.device)
                    if layer_i  > self.model.pruning_loc[0]:
                        policy[:, v_token_start:text_token_start] = pred_score_vis.type(dtype = self.dtype)
                    policy_list.append(policy)
            else:
                if not shared_policy:
                    for layer_i in range(len(full_outputs['attentions'])):
                        image_shape = 576
                        v_token_start = self.pre_prompt_length_list[0] if len(self.pre_prompt_length_list) != 0 else 0
                        text_token_start = v_token_start + image_shape
                        L_input = full_outputs['attentions'][layer_i].shape[-1]
                        B = full_outputs['attentions'][layer_i].shape[0]
                        policy = torch.ones(B, L_input, dtype=full_outputs['attentions'][layer_i].dtype, device=self.device)
                        if layer_i  > self.model.pruning_loc[0]:
                            num_to_zero = int((text_token_start - v_token_start) * mask_rate)
                            indices = (torch.randperm(int(text_token_start - v_token_start),device=policy.device)[:num_to_zero] + v_token_start ).to(policy.device)
                            policy[:, indices] = 0
                        policy_list.append(policy)
                else:
                    image_shape = 576
                    v_token_start = self.pre_prompt_length_list[0] if len(self.pre_prompt_length_list) != 0 else 0
                    text_token_start = v_token_start + image_shape
                    num_layers = len(full_outputs['attentions'])

                    device = self.device
                    dtype = full_outputs['attentions'][0].dtype

                    policy_list = [None] * num_layers

                    base_policy = torch.ones(full_outputs['attentions'][0].shape[0], full_outputs['attentions'][0].shape[-1], dtype=dtype, device=device)

                    num_to_zero = int((text_token_start - v_token_start) * mask_rate)

                    indices = torch.randperm(int(text_token_start - v_token_start), device=device)[:num_to_zero] + v_token_start

                    for layer_i in range(num_layers):
                        if layer_i > self.model.pruning_loc[0]:
                            policy = base_policy.clone()
                            policy[:, indices] = 0
                        else:
                            policy = base_policy

                        policy_list[layer_i] = policy

            recycle_list = True
            if recycle_list:
                if shared_policy:
                    sparse_key_values = process_kv_cache_shared(full_outputs['past_key_values'], policy_list)

                    vasparse_key_values = replace_kv_prefix_shared(outputs['past_key_values'], sparse_key_values, prefilling_prompt_len_list)

                    vasparse_model_kwargs = self._update_model_kwargs_for_vasparse_contrastive_decoding(outputs, vasparse_key_values, model_kwargs)
                    
                    vasparse_model_inputs = self.prepare_inputs_for_generation(input_ids, **vasparse_model_kwargs)
                else:
                    sparse_key_values = process_kv_cache(full_outputs['past_key_values'], policy_list)

                    vasparse_key_values = replace_kv_prefix(outputs['past_key_values'], sparse_key_values, prefilling_prompt_len_list)

                    vasparse_model_kwargs = self._update_model_kwargs_for_vasparse_contrastive_decoding(outputs, vasparse_key_values, model_kwargs)
                    
                    vasparse_model_inputs = self.prepare_inputs_for_generation(input_ids, **vasparse_model_kwargs)

                findings_kwargs['vasparse_outputs_base_layer'] = True
                findings_kwargs['base_contrastive_layer'] = base_contrastive_layer
                sparse_early_exit_layers = [base_contrastive_layer]
                vasparse_dict_outputs, vasparse_outputs = self(
                    **vasparse_model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    early_exit_layers=sparse_early_exit_layers,
                    findings_kwargs = findings_kwargs,
                )
                findings_kwargs['vasparse_outputs_base_layer'] = False


                ag_final_logits = vasparse_dict_outputs[base_contrastive_layer][:, -1, :]

            if  base_layer_not_sure:
                base_logits = dict_outputs[base_layer][:, -1, :]
                final_logits = dict_outputs[mature_layer][:, -1, :]
                if relative_top > 0.0:
                    final_logits = relative_top_filter(final_logits, relative_top)
                    base_logits = base_logits.log_softmax(dim=-1)
                    mask = final_logits[0] < -1e3
                    base_logits[0][mask] = -1e3

                logits = final_logits - base_logits
                next_token_logits = logits
            else:
                next_token_logits = dict_outputs[mature_layer][:, -1, :]

            next_token_logits = next_token_logits - contrastive_rate* ag_final_logits
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len) #torch.Size([2, 32000])
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size) #torch.Size([2, 32000])
            
        else:

            next_token_logits = dict_outputs[32][:, -1, :]
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len) #torch.Size([2, 32000])
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size) #torch.Size([2, 32000])
            

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)


        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size) # error torch.Size([2, 32000]) ->(2, 2*32000)

        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

        if return_dict_in_generate and output_scores:
            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

    sequence_outputs = beam_scorer.finalize(
        input_ids,
        beam_scores,
        next_tokens,
        next_indices,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_length=stopping_criteria.max_length,
        beam_indices=beam_indices,
    )

    end_time = time.perf_counter()
    total_time = total_time + (end_time - start_time)
    print("avg time per token: ", total_time / total_count_token)
    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if self.config.is_encoder_decoder:
            return BeamSearchEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return BeamSearchDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return sequence_outputs["sequences"]

