# transformers, pytorch
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

label_dict = {
    "title": 1,
    "bg": 2,
    # "obj": 3, # objはbgに統合
    "obj": 2,
    "method": 4,
    "res": 5,
    "other": 6,
}
num_label_dict = {
    "1": "title",
    "2": "bg",
    "3": "obj",
    "4": "method",
    "5": "res",
    "6": "other"
}

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    # '': get_constant_schedule,             # not supported for now
    # '': get_constant_schedule_with_warmup, # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"
tokenizer_name = "allenai/scibert_scivocab_cased"