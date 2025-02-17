"""
Helpers for probing large language model prediction probabilities
"""

import numpy as np
import torch
from torch.nn.functional import softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    raise Exception("Change runtime type to include a GPU.")


def probe_t5(model, tokenizer, target_id, context, verbose=False):
    # tokenize context
    input_ids = tokenizer(
        context,
        padding="longest",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    # use model to solicit a prediction
    outputs = model.generate(
        input_ids=input_ids.to(device),
        output_scores=True,
        return_dict=True,
        return_dict_in_generate=True,
        max_new_tokens=4,
    )

    # find the left-most non-sepecial token, save itr of this token to grab
    # correct logit scores array
    sequences = outputs["sequences"][0].tolist()
    for i in range(4):
        logits = outputs["scores"][i]
        probs = softmax(logits, dim=-1)
        probs = probs.detach().cpu().numpy()
        if tokenizer.decode([np.argmax(probs)]) not in [
            "<extra_id_0>",
            "",
            " ",
            "<pad>",
        ]:
            save_itr = i
            break
    # grab its logits
    logits = outputs["scores"][save_itr]
    # convert our prediction scores to a probability distribution with softmax
    probs = softmax(logits, dim=-1)
    probs = probs.detach().cpu().numpy()
    # grab the full decoded output for verbose:
    decoded_output = tokenizer.decode(sequences)

    if verbose:
        print(f"\n\tcontext... {context}")
        print(f"\ttokenized_context ids... {input_ids}")
        print(f"\tdecoded tokenized_context... {tokenizer.decode(input_ids[0])}")
        print(f"\tdecoded target id... {tokenizer.decode([target_id.item()])}")
        print(
            f"\tmost probable prediction id decoded... {tokenizer.decode([np.argmax(probs)])}"
        )
        print(f"\tdecoded full text generate output... {decoded_output}\n")
    return probs[0][target_id.item()]


def probe_stablelm(model, tokenizer, target_id, context, verbose=False):
    # tokenize context
    input_ids = tokenizer(
        context,
        padding="longest",
        max_length=4096,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    # use model to solicit a prediction
    outputs = model.generate(
        input_ids=input_ids.to(device),
        output_scores=True,
        return_dict=True,
        return_dict_in_generate=True,
        max_new_tokens=4,
    )

    # find the left-most non-sepecial token, save itr of this token to grab
    # correct logit scores array
    sequences = outputs["sequences"][0].tolist()
    for i in range(4):
        logits = outputs["scores"][i]
        probs = softmax(logits, dim=-1)
        probs = probs.detach().cpu().numpy()
        if tokenizer.decode([np.argmax(probs)]) not in [
            "<|endoftext|>",
            "<|padding|>",
            "",
            " ",
        ]:
            save_itr = i
            break
    # grab its logits
    logits = outputs["scores"][save_itr]
    # convert our prediction scores to a probability distribution with softmax
    probs = softmax(logits, dim=-1)
    probs = probs.detach().cpu().numpy()
    # grab the full decoded output for verbose:
    decoded_output = tokenizer.decode(sequences)

    if verbose:
        print(f"\n\tcontext... {context}")
        print(f"\ttokenized_context ids... {input_ids}")
        print(f"\tdecoded tokenized_context... {tokenizer.decode(input_ids[0])}")
        print(f"\tdecoded target id... {tokenizer.decode([target_id.item()])}")
        print(
            f"\tmost probable prediction id decoded... {tokenizer.decode([np.argmax(probs)])}"
        )
        print(f"\tdecoded full text generate output... {decoded_output}\n")
    return probs[0][target_id.item()]


def probe_falcon(model, tokenizer, target_id, context, verbose=False):
    # tokenize context
    input_ids = tokenizer(
        context,
        padding="longest",
        max_length=2048,
        truncation=True,
        return_token_type_ids=False,
        return_tensors="pt",
    ).input_ids
    # use model to solicit a prediction
    outputs = model.generate(
        input_ids=input_ids.to(device),
        output_scores=True,
        return_dict=True,
        return_dict_in_generate=True,
        max_new_tokens=3,
    )

    # find the left-most non-sepecial token, save itr of this token to grab
    # correct logit scores array
    sequences = outputs["sequences"][0].tolist()
    for i in range(3):
        logits = outputs["scores"][i]
        probs = softmax(logits, dim=-1)
        probs = probs.detach().cpu().numpy()
        if tokenizer.decode([np.argmax(probs)]) not in [
            "<|endoftext|>",
            "<|padding|>",
            "",
            " ",
        ]:
            save_itr = i
            break
    # grab its logits
    logits = outputs["scores"][save_itr]
    # convert our prediction scores to a probability distribution with softmax
    probs = softmax(logits, dim=-1)
    probs = probs.detach().cpu().numpy()
    # grab the full decoded output for verbose:
    decoded_output = tokenizer.decode(sequences)

    if verbose:
        print(f"\n\tcontext... {context}")
        print(f"\ttokenized_context ids... {input_ids}")
        print(f"\tdecoded tokenized_context... {tokenizer.decode(input_ids[0])}")
        print(f"\tdecoded target id... {tokenizer.decode([target_id.item()])}")
        print(
            f"\tmost probable prediction id decoded... {tokenizer.decode([np.argmax(probs)])}"
        )
        print(f"\tdecoded full text generate output... {decoded_output}\n")
    return probs[0][target_id.item()]


def probe_redpajama(model, tokenizer, target_id, context, verbose=False):
    # tokenize context
    input_ids = tokenizer(
        context,
        padding="longest",
        max_length=2048,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    # use model to solicit a prediction
    outputs = model.generate(
        input_ids=input_ids.to(device),
        output_scores=True,
        return_dict=True,
        return_dict_in_generate=True,
        max_new_tokens=4,
    )

    # find the left-most non-sepecial token, save itr of this token to grab
    # correct logit scores array
    sequences = outputs["sequences"][0].tolist()
    for i in range(4):
        logits = outputs["scores"][i]
        probs = softmax(logits, dim=-1)
        probs = probs.detach().cpu().numpy()
        if tokenizer.decode([np.argmax(probs)]) not in [
            "<|endoftext|>",
            "<|padding|>",
            "",
            " ",
        ]:
            save_itr = i
            break
    # grab its logits
    logits = outputs["scores"][save_itr]
    # convert our prediction scores to a probability distribution with softmax
    probs = softmax(logits, dim=-1)
    probs = probs.detach().cpu().numpy()
    # grab the full decoded output for verbose:
    decoded_output = tokenizer.decode(sequences)

    if verbose:
        print(f"\n\tcontext... {context}")
        print(f"\ttokenized_context ids... {input_ids}")
        print(f"\tdecoded tokenized_context... {tokenizer.decode(input_ids[0])}")
        print(f"\tdecoded target id... {tokenizer.decode([target_id.item()])}")
        print(
            f"\tmost probable prediction id decoded... {tokenizer.decode([np.argmax(probs)])}"
        )
        print(f"\tdecoded full text generate output... {decoded_output}\n")
    return probs[0][target_id.item()]


def probe_mpt(model, tokenizer, target_id, context, verbose=False):
    # tokenize context
    input_ids = tokenizer(
        context,
        padding="longest",
        max_length=2048,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    # use model to solicit a prediction
    outputs = model.generate(
        input_ids=input_ids.to(device),
        output_scores=True,
        return_dict=True,
        return_dict_in_generate=True,
        max_new_tokens=4,
    )

    # find the left-most non-sepecial token, save itr of this token to grab
    # correct logit scores array
    sequences = outputs["sequences"][0].tolist()
    for i in range(4):
        logits = outputs["scores"][i]
        probs = softmax(logits, dim=-1)
        probs = probs.detach().cpu().numpy()
        if tokenizer.decode([np.argmax(probs)]) not in [
            "<|endoftext|>",
            "<|padding|>",
            "",
            " ",
        ]:
            save_itr = i
            break
    # grab its logits
    logits = outputs["scores"][save_itr]
    # convert our prediction scores to a probability distribution with softmax
    probs = softmax(logits, dim=-1)
    probs = probs.detach().cpu().numpy()
    # grab the full decoded output for verbose:
    decoded_output = tokenizer.decode(sequences)

    if verbose:
        print(f"\n\tcontext... {context}")
        print(f"\ttokenized_context ids... {input_ids}")
        print(f"\tdecoded tokenized_context... {tokenizer.decode(input_ids[0])}")
        print(f"\tdecoded target id... {tokenizer.decode([target_id.item()])}")
        print(
            f"\tmost probable prediction id decoded... {tokenizer.decode([np.argmax(probs)])}"
        )
        print(f"\tdecoded full text generate output... {decoded_output}\n")
    return probs[0][target_id.item()]


def probe_gpt(model, tokenizer, target_id, context, verbose=False):
    # tokenize context
    input_ids = tokenizer(
        context,
        return_tensors="pt",
    ).input_ids.to(device)

    # grab value
    target_scalar = target_id.detach().cpu().numpy()

    # use model to solicit a prediction
    outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)

    # every token in the model's vocab gets a representative prediction from the model
    logits = outputs["logits"][0, -1]
    # convert our prediction scores to a probability distribution with softmax
    probs = softmax(logits, dim=-1)
    probs = list(probs.detach().cpu().numpy())

    if verbose:
        print(f"\n\tcontext... {context}")
        print(f"\ttokenized_context ids... {input_ids}")
        print(f"\tdecoded tokenized_context... {tokenizer.decode(input_ids[0])}")
        print(f"\tdecoded target id... {tokenizer.decode([target_id.item()])}")
        print(
            f"\tmost probable prediction id decoded... {tokenizer.decode([np.argmax(probs)])}\n"
        )

    # double check weird-ness before accessing prob
    if len(probs) < target_id:
        return None

    # return the likelihood that our stipulated target would follow the context,
    # according to the model
    try:
        return np.take(probs, [target_scalar])[0]

    except IndexError:
        print("target index not in model vocabulary scope; raising IndexError")
        return None


def probe_bert(model, tokenizer, target_id, context, verbose=False):
    # tokenize context
    input_ids = tokenizer(
        context,
        padding="longest",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).input_ids

    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    # use model to solicit a prediction
    logits = model(input_ids=input_ids.to(device)).logits
    mask_token_logits = logits[0, mask_token_index, :]

    # Convert our prediction scores to a probability distribution with softmax
    probs = torch.squeeze(softmax(mask_token_logits, dim=-1))

    probs = probs.detach().cpu().numpy()

    if verbose:
        print(f"\n\tcontext... {context}")
        print(f"\ttokenized_context ids... {input_ids}")
        print(f"\tdecoded tokenize_context... {tokenizer.decode(input_ids[0])}")
        print(f"\tmask token id... {tokenizer.mask_token_id}")
        print(f"\tmask token index in context... {mask_token_index}")
        print(f"\tdecoded target id... {tokenizer.decode([target_id.item()])}")
        print(
            f"\tmost probable prediction id decoded... {tokenizer.decode([np.argmax(probs)])}\n"
        )

    return probs[target_id.item()]


def probe_llama(model, tokenizer, target_id, context, verbose=False):
    # tokenize context
    input_ids = tokenizer(
        context,
        return_tensors="pt",
    ).input_ids.to(device)

    # grab value
    target_scalar = target_id.detach().cpu().numpy()

    # use model to solicit a prediction
    outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)

    # every token in the model's vocab gets a representative prediction from the model
    logits = outputs["logits"][0, -1]
    # convert our prediction scores to a probability distribution with softmax
    probs = softmax(logits, dim=-1)

    probs = list(probs.detach().cpu().numpy())

    if verbose:
        print(f"\n\tcontext... {context}")
        print(f"\ttokenized_context ids... {input_ids}")
        print(f"\tdecoded tokenized_context... {tokenizer.decode(input_ids[0])}")
        print(f"\tdecoded target id... {tokenizer.decode([target_id.item()])}")
        print(
            f"\tmost probable prediction id decoded... {tokenizer.decode([np.argmax(probs)])}\n"
        )

    # double check weird-ness before accessing prob
    if len(probs) < target_id:
        return None

    # return the likelihood that our stipulated target would follow the context,
    # according to the model
    try:
        return np.take(probs, [target_scalar])[0]

    except IndexError:
        print("target index not in model vocabulary scope; raising IndexError")
        return None
