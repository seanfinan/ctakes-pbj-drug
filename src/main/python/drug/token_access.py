from itertools import product

from main_folder.ctakes_types import *


def get_args_to_rel_map(cas):
    args_to_rel = {}
    for bin_text_rel in cas.select(BinaryTextRelation):
        # thanks to DeepPheAnaforaXMLReader.java
        # arg 1 is guaranteed to be med
        # arg 2 - " - sig
        med_arg = bin_text_rel.arg1.argument
        sig_arg = bin_text_rel.arg2.argument
        args_to_rel[(med_arg, sig_arg)] = bin_text_rel
    return args_to_rel


def ctakes_tokenize(cas, sentence):
    return sorted(cas.select_covered(BaseToken, sentence), key=lambda t: t.begin)


def ctakes_clean(cas, sentence):
    base_tokens = []
    token_map = []
    newline_tokens = cas.select_covered(NewlineToken, sentence)
    newline_token_indices = {(item.begin, item.end) for item in newline_tokens}

    for base_token in ctakes_tokenize(cas, sentence):
        if (base_token.begin, base_token.end) not in newline_token_indices:
            base_tokens.append(base_token.get_covered_text())
            token_map.append((base_token.begin, base_token.end))
        else:
            base_tokens.append('<cr>')
    return " ".join(base_tokens), token_map


def get_relex_labels(cas, sentences):
    doc_labels = []
    args_to_rel = get_args_to_rel_map(cas)
    max_sent_len = 0
    for sentence in sentences:
        sent_labels = []
        med_mentions = cas.select_covered(MedicationMention, sentence)
        sig_mentions = cas.select_covered(EntityMention, sentence)
        newline_tokens = cas.select_covered(NewlineToken, sentence)
        newline_token_indices = {(item.begin, item.end) for item in newline_tokens}

        token_start_position_map = {}
        curr_token_idx = 0
        base_tokens = []

        for base_token in ctakes_tokenize(cas, sentence):
            if (base_token.begin, base_token.end) not in newline_token_indices:
                if base_token.begin in token_start_position_map.keys():
                    ValueError("Two tokens start with the same index")
                token_start_position_map[base_token.begin] = curr_token_idx
                base_tokens.append(base_token.get_covered_text())
            else:
                base_tokens.append('<cr>')
            curr_token_idx += 1

        if med_mentions and sig_mentions:
            for med_sig_pair in product(med_mentions, sig_mentions):
                if med_sig_pair in args_to_rel.keys():
                    med_sig_rel = args_to_rel[med_sig_pair]
                    label = med_sig_rel.category
                    # else:
                    # label = 'None'
                    med_arg, sig_arg = med_sig_pair
                    if med_arg.begin in token_start_position_map and sig_arg.begin in token_start_position_map:
                        med_idx = token_start_position_map[med_arg.begin]
                        sig_idx = token_start_position_map[sig_arg.begin]
                        sent_labels.append((med_idx, sig_idx, label))
        else:
            sent_labels = 'None'
        sent_len = curr_token_idx
        if sent_len > max_sent_len:
            max_sent_len = sent_len
        # print(f"{sent_labels} : {' '.join(base_tokens)}")
        doc_labels.append(sent_labels)
    return doc_labels, max_sent_len
