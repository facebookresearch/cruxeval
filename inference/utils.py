# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
import warnings
from collections import defaultdict

import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially. See compute_code for more details.
    The prompt can either be:
    - one prompt: normal code completion
    - two prompts: for infilling mode (prefix, suffix) or instructin-tuning mode (instruction, context)
    """

    def __init__(
        self,
        task,
        dataset,
        tokenizer,
        max_length,
        n_tasks=None,
        n_copies=1,
        prefix="",
    ):
        self.task = task
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_tasks = n_tasks
        self.n_copies = n_copies
        self.prefix = prefix

    def __iter__(self):
        prompts = []
        row_idxs = []
        for sample in range(self.n_tasks):
            dataset_sample = self.dataset[sample]
            prompt_contents = self.task.get_prompt(dataset_sample)
            assert isinstance(prompt_contents, str)
            prompt = self.prefix + prompt_contents
            prompts.append(prompt)
            row_idxs.append(dataset_sample["row_index"])

        return_token_type_ids = None  # default

        outputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            return_token_type_ids=return_token_type_ids,
        )

        for sample in range(self.n_tasks):
            for _ in range(self.n_copies):
                yield {
                    "row_index": row_idxs[sample],
                    "prompt": prompts[sample],
                    "ids": outputs.input_ids[sample],
                    "input_len": outputs.attention_mask[sample].sum(),
                }


def complete_code(
    task,
    model,
    sampling_params,
    dataloader,
    batch_size,
    n_tasks,
    prefix="",
    postprocess=True,
    backend='vllm',
    tokenizer=None,
):
    max_length_generation = sampling_params.max_tokens
    code_gens = defaultdict(list)
    code_gens_raw = defaultdict(list)
    total = math.ceil(n_tasks * dataloader.dataset.n_copies)
    for step, batch in tqdm(enumerate(dataloader), total=total):
        inputs = batch["ids"][:, : batch["input_len"]].tolist()
        num_tokens = len(inputs[0])
        if max_length_generation - num_tokens < 0:
            code_gens[int(batch["row_index"][0])].extend([""] * batch_size)
            code_gens_raw[int(batch["row_index"][0])].extend([""] * batch_size)
            warnings.warn(
                f"Skipping task {batch['row_index'][0]} because it is too long -- [{max_length_generation=}|{num_tokens=}]"
            )
            continue
        sampling_params.max_tokens = max_length_generation - num_tokens

        generated_tasks = batch["row_index"].repeat(batch_size)
        if backend == 'vllm':
            outputs = model.generate(
                prompt_token_ids=inputs, sampling_params=sampling_params, use_tqdm=False
            )
            generated_texts = [o.text for o in outputs[0].outputs]
        elif backend == 'gptqmodel':
            inputs_tensor = torch.tensor(inputs).to(model.device)
            model_kwargs = {
                "input_ids": inputs_tensor,
                "max_new_tokens": sampling_params.max_tokens,
                "num_return_sequences": sampling_params.n,
            }

            # normalize stop_strings
            if sampling_params.stop is not None:
                if isinstance(sampling_params.stop, str):
                    sampling_params.stop = [sampling_params.stop]

                # transformer require tokenizer to be passed when stop_strings is used
                model_kwargs["stop_strings"] = sampling_params.stop
                model_kwargs["tokenizer"] = tokenizer

            do_sample = True if sampling_params.temperature != 1.0 else False

            if do_sample:
                model_kwargs["temperature"] = sampling_params.temperature
                model_kwargs["do_sample"] = do_sample

            if sampling_params.top_k > 0:
                model_kwargs["top_k"] = sampling_params.top_k

            if sampling_params.top_p != 1.0:
                model_kwargs["top_p"] = sampling_params.top_p

            outputs = model.generate(**model_kwargs)
            generated_texts = tokenizer.batch_decode(
                outputs[:, inputs_tensor.size(-1):],
                skip_special_tokens=True,
            )

            # transformers will inlcude stop_string in output, normalize (strip) to align with vllm default
            if sampling_params.stop is not None:
                for stop_string in sampling_params.stop:
                    generated_texts = [generated_text.replace(stop_string, "") for generated_text in generated_texts]
        else:
            raise ValueError("backend support: [vllm, gptqmodel]")

        combined_texts = [
            batch["prompt"][0] + generated_text for generated_text in generated_texts
        ]

        for task_idx, text in zip(generated_tasks, combined_texts):
            task_idx = int(task_idx.item())
            if postprocess:
                text_processed = task.postprocess_generation(text, task_idx)
            code_gens[task_idx].append(text_processed)
            code_gens_raw[task_idx].append(text)

    return code_gens, code_gens_raw
