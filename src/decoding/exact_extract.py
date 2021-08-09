import dataclasses
from functools import reduce
import torch
import transformers
from torch import Tensor

from src.utils.outputs import ExtractionResults, PredictionWithMetadata, AggregatedPredictionsOfType, SpanType


@dataclasses.dataclass
class InputAndAttention:
    input_ids: Tensor
    attention_mask: Tensor

def expand_context(context, bos_token_id, device):
    """Get a N-length context (passage) tokens and return a (N,N) tensor of all its rolls (in-order)

    This can be used when preparing the decoder input/output representing every predix of the context. An additional
    left column is added with the bos_token_id, since for T5 it is assumed that each string would begin with
    <extrac_id_0>, but can be used for other models with another BOS token.

    :param context: A 1-D or (N,1) shaped token ids tensor of the passage.
    :param bos_token_id: Token to be used as BOS for all rolls. If the model does not require a BOS, need to overwrite
    the method.
    :param device: The device to move the context to.
    :return: A (N,N) tensor with all the rolls of the context.
    """
    if len(context.shape) == 1:
        context = context.view(1, -1)
    rolls = []
    for i in range(context.shape[-1] - 1):
        rolls.append(context.roll(-1 * i)[0])
    context = torch.stack(rolls)
    bos_column = torch.ones(context.shape[0], 1, dtype=torch.long) * bos_token_id
    if device:
            pad_column = bos_column.to(device)
    context = torch.cat((pad_column, context), dim=1)
    return context


def prepare_nlls_infs(context, device):
    """Build a inf-scores mask to rule out the scores in irrelevant positions of the scores matrix.

    Prepare a mask of infs below the main diagonal and zeros o.w. This is needed since the nll score of location (i,j)
    where j>i are of non-spans, since it ends with tokens warped from the begging of the passage.

    :param context: The context (passage), used to extract the shape, should be (N,1).
    :param device: The device to move the mask to.
    :return: (N,N) tensor used as nlls inf-scores mask.
    """
    nlls_infs = torch.ones((context.shape[0], context.shape[1]), dtype=torch.float).to(device).tril().fliplr()
    nlls_infs = nlls_infs.masked_fill(nlls_infs != 0, float('inf'))
    nlls_infs[:, 0] = float('inf')  # the first index predicts probability for jst mask
    return nlls_infs


def prepare_ml_span_inputs(model: transformers.EncoderDecoderModel, tokenizer: transformers.T5TokenizerFast,
                           encoder_input_attention: InputAndAttention, context: Tensor, chunk_size: int, device: int,
                           trim_context: int):
    """Prepare the inputs for calculating the most likely span. Do a forward pass on the encoder using the input and
    expand the context (passage) to a matrix holding all its suffices, to later be used as contextualization for all
    passage suffixes. Prepare the matching nll scores mask, see the "prepare_nlls_infs" method for elaboration.
    Return "encoder_outputs" as the encoder last hidden state duplicated to the chunk size shape. If the total context
    size is not divisible in the context size the returned "encoder_last_hidden_state" can be used later.

    :param model: The encoder-decoder model, used for performing an encoder forward pass once for every
    :param tokenizer: The tokenizer at use.
    :param encoder_input_attention: dataclass holding the input tokens and attention mask.
    :param context: The passage to be used for finding most likely span from.
    :param chunk_size: Go over the context (passage) suffixes in chunks to avoid memory issues.
    :param device: gpu index to be used (only single device supported).
    :param trim_context: Only allow answers of this size to be extracted; larger chunks can be used hence improved
    performace on the expanse of ignoring longer answers.
    :return: a tuple of (expanded_context, nlls_infs, encoder_outputs, encoder_last_hidden_state, attention_mask)
    """
    chunk_size = min(chunk_size, context.shape[0]-1)
    bos_token_id = tokenizer.additional_special_tokens_ids[0]
    expanded_context = expand_context(context, bos_token_id, device)
    nlls_infs = prepare_nlls_infs(expanded_context, device)
    expanded_context = expanded_context[:, :trim_context]
    nlls_infs = nlls_infs[:, :trim_context]
    input_ids = encoder_input_attention.input_ids.view(1, -1)
    attention_mask = encoder_input_attention.attention_mask.view(1, -1)
    encoder_last_hidden_state = None
    while encoder_last_hidden_state is None and input_ids.shape[-1] > 0:
        try:
            encoder_last_hidden_state = model.encoder.forward(input_ids=input_ids,
                                                              attention_mask=attention_mask).last_hidden_state
        except RuntimeError as e:
            print(e)
            print(f'input of size {input_ids.shape} failed to pass encoder, reducing to {input_ids[:, 10:].shape}')
            input_ids = input_ids[:, 10:]
    encoder_outputs = (None, torch.cat([encoder_last_hidden_state] * chunk_size), None)
    return expanded_context, nlls_infs, encoder_outputs, encoder_last_hidden_state, attention_mask

def mlspan(model: transformers.EncoderDecoderModel, tokenizer: transformers.T5TokenizerFast,
           encoder_input_attention: InputAndAttention, context: Tensor, chunk_size: int, device: int, trim_context: int,
           eos_token_id: int) -> ExtractionResults:
    """Main entry point. Iterate the context (passage) by chunks, extract most likely span from each and argmax all.

    :param model: encoder-decoder model used for calculating probability of spans.
    :param tokenizer: used for extracting the <extra_id_0> token id.
    :param encoder_input_attention: input ids and attention mask for the input; should be the passage and question.
    :param context: The passage from which to extract (although it can be identical to encoder ids, it should be the
    passge only, without the qestion and pattern parts.
    :param chunk_size: chunk size when iterating over the passage suffixes. lower => slower yet less memory.
    :param device: device id to work on, needed for loading manually created tensors. should've been taken from model..
    :param trim_context: Only allow answers of this size to be extracted; larger chunks can be used hence improved
    performace on the expanse of ignoring longer answers.
    :param eos_token_id: token to be used for eos, for T5 should be <extra_id_1>.
    :return: ExtractionResult with most likely span from the passge, both in the length-nomalized and the non-normalized
    versions.
    """
    expanded_context, nlls_infs, encoder_outputs, encoder_last_hidden_state, attention_mask = \
        prepare_ml_span_inputs(
        model, tokenizer, encoder_input_attention, context, chunk_size, device, trim_context)
    # calculate in chunks
    chunk_results = []
    for i in range(0, expanded_context.shape[0], chunk_size):
        context_chunk = expanded_context[i:i + chunk_size + 1, :]
        nlls_infs_chunk = nlls_infs[i:i + chunk_size + 1, :]
        if encoder_outputs[1].shape[0] != context_chunk.shape[0]:  # might happen on last chunk
            encoder_outputs = (None, torch.cat([encoder_last_hidden_state] * context_chunk.shape[0]), None)
        chunk_result = mlspan_calc(model=model, tokenizer=tokenizer, encoder_outputs=encoder_outputs,
                                   attention_mask=attention_mask, context=context_chunk,
                                   nlls_infs=nlls_infs_chunk, device=device, eos_token_id=eos_token_id)
        chunk_results.append(chunk_result)
    result = reduce(lambda a, b: a.merge(b), chunk_results)
    result.context_tokens = tokenizer.convert_ids_to_tokens(context[:-1].tolist())
    return result

def extract_tops(tokenizer, context, prfx_nll, tkn_nlls, relevant_eos_nlls, eos_token_id):
    """Return a result containing the most likely span alongside its tokens nlls.

    :param tokenizer: Tokenizer used to decode the most likely span.
    :param context: the passage (tokenized) used to extract the span from (nll matrix provide indices)
    :param prfx_nll: the prefix-of-suffixes cumsum nlls ( L(.,.) in the paper ).
    :param tkn_nlls: per-token nll to be extracted for the best answers per-token nll ( l(.,.) in the paper )
    :param relevant_eos_nlls: the eos nlls per position, used to get eos nll of selected span ( e(.,.) in the paper)
    :param eos_token_id: The EOS token id used later to append to the best span when returning its tokens.
    :return: Result containing the most likely span alongside its tokens and nlls.
    """
    top_ind = [(prfx_nll == torch.min(prfx_nll)).nonzero()[0]][0]
    r, c = tuple(top_ind.cpu().numpy())
    pred: PredictionWithMetadata = extract_tokens_and_probs_from_nlls(tokenizer=tokenizer, context=context,
                                                                      prefix_nlls=prfx_nll, tokens_nlls=tkn_nlls,
                                                                      eos_nlls=relevant_eos_nlls, r=r, c=c,
                                                                      eos_token_id=eos_token_id)
    return [pred]


def extract_tokens_and_probs_from_nlls(tokenizer, context, prefix_nlls, tokens_nlls, eos_nlls, r, c,
                                       eos_token_id) -> PredictionWithMetadata:
    """Given nll scores and indices of best start  index (r) and length (c), extract tokens and scores into a result.
    See "extract_tops" documentation for explanation

    :param tokenizer: Tokenizer used to decode the most likely span.
    :param context: the passage (tokenized) used to extract the span from.
    :param prfx_nll: the prefix-of-suffixes cumsum nlls ( L(.,.) in the paper ).
    :param tkn_nlls: per-token nll to be extracted for the best answers per-token nll ( l(.,.) in the paper )
    :param eos_nlls: the eos nlls per position, used to summarize with the prfx_nlls ( e(.,.) in the paper)
    :param r: row in prefix-nlls (aka context suffix index) of the to-be-extracted span.
    :param c: columns in prefix-nlls (aka predix length of context suffix) of the to-be-extracted span.
    :param eos_token_id: The EOS token id used later to append to the best span when returning its tokens.
    :return: Result containing the span at position (c,r) alongside its tokens and nlls.
    """
    tokens_ids = context[r, :c + 1].cpu().tolist() + [eos_token_id]
    tokens = [tokenizer._convert_id_to_token(id) for id in tokens_ids]
    token_nlls = [x.item() for x in tokens_nlls[r, :c + 1]] + [eos_nlls[r, c].item()]
    nll = prefix_nlls[r, c].item()
    return PredictionWithMetadata(tokens=tokens, tokens_nlls=token_nlls,
                                  decoded=tokenizer.decode(tokens_ids, skip_special_tokens=True), nll=nll,
                                  tokens_ids=tokens_ids)


def mlspan_calc(model, tokenizer, encoder_outputs, attention_mask, context, nlls_infs, device,
                eos_token_id) -> ExtractionResults:
    """Main method that calculated most likely span, retrieves an ExtractionResult.

    :param model: encoder-decoder model used for calculating probability of spans.
    :param tokenizer: used for extracting the <extra_id_0> token id.
    :param encoder_outputs: The encoder output (repeated to match the batch size).
    :param attention_mask: attention mask over encoder for cross attention.
    :param context: The passage from which to extract (although it can be identical to encoder ids, it should be the
    :param nlls_infs: inf-mask for all invalid spans indices.
    :param device: Devive to move tensors to.
    :param eos_token_id: EOS token of tokenizer, used as the last predicted token for each span.
    :return: ExtractionResults of results for both most likely span in both regualar and length-norm versions.
    """
    decoder_input = model._shift_right(context)
    logits = model.forward(encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input,
                           attention_mask=attention_mask).logits
    # l(.,.) in the paper
    tokens_nlls = torch.nn.CrossEntropyLoss(reduction='none')(input=logits.permute(0, 2, 1), target=context)
    eoss = (torch.ones_like(context) * eos_token_id).to(device)
    # e(.,.) in the paper
    eos_nlls = torch.nn.CrossEntropyLoss(reduction='none')(input=logits.permute(0, 2, 1), target=eoss)
    eos_nlls = eos_nlls.roll(-1, 1)
    # L + e in the paper
    prefixes_nlls = tokens_nlls.cumsum(dim=1) + eos_nlls
    # L + e in the paper plus masking invalid spans indices (warped spans)
    nlls = prefixes_nlls + nlls_infs
    # Making length normalized version, on very scarce cases was better then nlls
    nlls_norm = nlls.div(torch.arange(start=2, end=nlls.shape[1] + 2).to(device))

    ml_span = AggregatedPredictionsOfType(
        extract_tops(tokenizer=tokenizer, context=context, prfx_nll=nlls, tkn_nlls=tokens_nlls,
                     relevant_eos_nlls=eos_nlls, eos_token_id=eos_token_id),
        SpanType.ML_SPAN, 1)
    ml_span_norm = AggregatedPredictionsOfType(
        extract_tops(tokenizer=tokenizer, context=context, prfx_nll=nlls_norm, tkn_nlls=tokens_nlls,
                     relevant_eos_nlls=eos_nlls, eos_token_id=eos_token_id),
        SpanType.ML_SPAN_LENGTH_NORMALIZED, 1)

    return ExtractionResults(
        greedy=None,
        beam=None,
        ml_span=ml_span,
        ml_span_norm=ml_span_norm,
        label=None,
        input=None,
        context_tokens=None
    )
