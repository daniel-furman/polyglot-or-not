import numpy as np

import torch
from torch.nn.functional import softmax


def probe_flan(model, input_ids, target):
    """
    model: a pretrained google model pulled in from HuggingFace (ie. flan-t5-small,
      flan-ul2, t5-small, etc.)
    input_ids: the indices (in the vocabulary) of our left-context tokens
    target: the index (in the vocabulary) of the token we're gathering a prediction for

    return: a float indicating the likelihood of the target following the left-context
      according to the model in case of error, return None
    """

    # Call the model
    outputs = model(
        input_ids=input_ids,
        decoder_input_ids=torch.tensor([[0, 32099]], device="cuda:0"),
        output_hidden_states=True,
        return_dict=True,
    )

    # We have batch size of 1, so grab that, then,
    # Take the entire last matrix which corresponds to the last layer
    logits = outputs["logits"][0, -1]

    # convert our prediction scores to a probability distribution with softmax
    probs = softmax(logits, dim=-1)

    probs = probs.detach().cpu().numpy()

    return probs[target.item()]


def probe_gpt2(model, input_ids, target):
    """
    model: a gpt pretrained model pulled in from HuggingFace
    input_ids: the indices in gpt's vocabulary of our left-context tokens
    target: the index in gpt's vocabulary of the token we're gathering a prediction for

    return: a float indicating the likelihood of the target following the left-context
      according to the model in case of error, return None
    """

    # ensure we're only asking for a single token prediction
    if len(target) > 1:
        # default to the very first token that get's predicted
        # e.g. in the case of Tokyo, which gets split into <Tok> <yo>,
        target = target[0]

    # sanity check - do a conversion that tells us the exact "token" predicted on
    # print(model.convert_)

    # grab value
    target_scalar = target.detach().cpu().numpy()

    # use model to solicit a prediction
    outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)

    # shape of 50257 which corresponds to the vocab size of GPT
    # every token in GPT's vocab gets a representative prediction from the model
    logits = outputs["logits"][0, -1]

    # convert our prediction scores to a probability distribution with softmax
    probs = softmax(logits, dim=-1)

    probs = list(probs.detach().cpu().numpy())

    # double check weird-ness before accessing prob
    if len(probs) < target:
        return None

    # return the likelihood that our stipulated target would follow the context,
    # according to the model
    try:
        return np.take(probs, [target_scalar])[0]

    except IndexError:

        print("target index not in model vocabulary scope; raising IndexError")
        return None
