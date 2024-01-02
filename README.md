# CRUXEval: Code Reasoning, Understanding, and Execution Evaluation

CRUXEval (**C**ode **R**easoning, **U**nderstanding, and e**X**ecution **Eval**uation) is a benchmark of 800 Python functions and input-output pairs. The benchmark consists of two tasks, CRUXEval-I (input prediction) and CRUXEval-O (output prediction). 

An example of a sample in the benchmark is shown here:
```python
def f(nums):
    count = len(nums)
    for i in range(-count+1, 0):
        nums.append(nums[i])
    return nums

# output prediction, CRUXEval-O
assert f([2, 6, 1, 3, 1]) == ??
# GPT-4: [2, 6, 1, 3, 1, 6, 1, 3, 1], incorrect

# input prediction, CRUXEval-I
assert f(??) == [2, 6, 1, 3, 1, 6, 3, 6, 6]
# GPT-4: [2, 6, 1], incorrect
```

The benchmark was constructed as follows: first, we use [Code Llama 34B](https://huggingface.co/codellama/CodeLlama-34b-hf) to generate a large set of functions and inputs. The outputs are generated by executing the functions on the inputs. Second, we filter the set so that our benchmark only consists of short problems with low computation and memory requirements, problems which a good human programmer should be able to do without extra memory in a minute or so. Third, we randomly select 800 samples passing the filter, ensuring the benchmark is both small enough to easily run but large enough to reliably see performance differences among various models.


## ⚙️ Setup and Installation
To clone the repository, run
```
git clone git@github.com:facebookresearch/cruxeval.git
cd cruxeval
```

## 📋 Requirements
If you want to install everything at once, run `pip install -r requirements.txt`. Otherwise, if you just want to score generations, run `pip install -r requirements-base.txt`. If you just want to run OpenAI models, run `pip install -r requirements-openai.txt`. If you just want to run inference on HuggingFace models, run `pip install -r requirements-inference.txt`. The code has been tested with Python version 3.9 and CUDA version 12.1.

## 🔥 Getting Started
The dataset is available in `.jsonl` format in `data/cruxeval.jsonl` and in [HuggingFace Datasets](https://huggingface.co/datasets/cruxeval-org/cruxeval). Each sample contains `code`, `input`, and `output` fields. A sample script to print the samples of the dataset is in `quickstart.ipynb`.

## 💯 Scoring Your Own Generations
To evaluate a set of generations, load your generations (function calls for CRUXEval-I or outputs for CRUXEval-O) as strings into a json file such as `generations.json` with the following format:
```
{
    "sample_0": ["f([1, 1, 1, 1, 3, 3])", "f([])"],
    ...
    "sample_799": ["f('~neqe-;ew22')", "f('~neqe-;ew22')"]
}
```

Then, `cd evaluation` and run the following command, setting `mode` to `input` to evaluate CRUXEval-I and `output` to evaluate CRUXEval-O.
```
python evaluate_generations.py \
    --generations_path generations.json \
    --scored_results_path generations_scored.json \
    --mode input
```

The script should take around a minute or so. An example of input and output generations in the correct format for Code Llama 7B can be found in the `model_generations` folder, and an example of the corresponding execution result file is in `evaluation/evaluation_results`. The execution results will be written to the file you specify in `--scored_results_path`. It contains `raw_generations` (the dictionary of raw generations for each sample that was provided), `raw_scored_generations` (the dictionary of scored results for each sample), and overall `pass_at_1` and `pass_at_5` scores. As an example to reproduce the scoring of Code Llama 7B CRUXEval-I generations, run the following command in the `evaluation` folder:
```
python3 evaluate_generations.py \
    --generations_path ../samples/model_generations/sample_codellama-7b_temp0.2_input/generations.json \
    --scored_results_path ../samples/evaluation_results/sample_scored_codellama-7b_temp0.2_input.json \
    --mode input
```
## ✅ Generated and Scored Outputs
We also open-source generations and outputs for the models we display on the leaderboard below. First, `cd samples`. To access the generations, run `unzip model_generations.zip`. To access the scored versions of the generations run `unzip evaluation_results.zip`. The generations and scored generations will appear in `samples/model_generations` and `samples/evaluation_results`, respectively.

## 🤖 Running Inference on HuggingFace Models
We provide a script compatible with SLURM to run inference on CRUXEval with HuggingFace models. First `cd inference`. Then, run `./scripts/run_input_prediction.sh` for CRUXEval-I or `./scripts/run_output_prediction.sh` for CRUXEval-O. The default script in the repository runs a variety of models with 2 GPU's at temperatures `0.2, 0.8` with `n_sample=10` generations per sample. You should change `--output, --error, --partition` accordingly and also may wish to change one or more of `GPUS, batch_size, n_samples, temperatures, dirs (directory names), models`.

This script parallelizes the 800 samples of the benchmark in a data-parallel fashion across the GPU's. After running the scripts, the generations will appear in `inference/model_generations_raw/shard_i.json`, where `i` ranges from `0` to `GPUS-1`. To convert these into a form that is readily available for evaluation, run `python combine_generations.py`, which will create a file `../model_generations/{MODEL_INFO}/generations.json`. The generations can then be evaluated by following the above instructions.

For best results, we recommend running WizardCoder with `transformers==4.31.0/vllm==0.1.4` and all other models with `transformers==4.34.1/vllm==0.2.2`. WizardCoder performance has been known to degrade with newer versions of transformers.

## 🤖 Running Inference on OpenAI Models
You need to use your own API key and comply with OpenAI terms of use. We provide a script to run inference on OpenAI models if you would like to try different temperatures or latest models. Set the `OPENAI_API_KEY` environmental variable to be your API key, for example via `export OPENAI_API_KEY = YOUR_KEY`. Then, `cd openai` and run `python openai_run.py`. Like before, the generations will appear in `../model_generations/{MODEL_INFO}/generations.json`.

## 💯 Scoring a Batch of Generations and Tabulating Results
Finally, we provide SLURM-based scripts to run evaluation on many models in parallel in `evaluation/evaluate_all_predictions_input.sh` and `evaluation/evaluate_all_predictions_output.sh`. You should change the `--output, --error, --partition` values and may also wish to change `run_names`. For convenience, we have provided a script `evaluation/print_evaluation_directories.py` that automatically prints all the directories found in `model_generations` to populate `run_names` with for both scripts.

All raw results (`raws`) and pass@1 and 5 scores (`pass@1` and `pass@5`) can then be found in the `evaluation/evaluation_results` folder. We have provided a script `evaluation/read_results.py` to print the results in tabular form.

## 🏆 Leaderboard

<center>
<table>
<tr><th>CRUXEval-I</th><th>CRUXEval-O</th></tr>
<tr> <td>

| Model                  |   Pass@1 |   Pass@5 |
|:----------------------:|:--------:|:--------:|
| phi-1                  |     13.1 |     21.1 |
| phi-1.5                |     23.2 |     37.7 |
| deepseek-instruct-1.3b |     27.2 |     40.1 |
| deepseek-base-1.3b     |     27.8 |     44.7 |
| starcoderbase-7b       |     29.7 |     47.3 |
| starcoderbase-16b      |     31.3 |     49.2 |
| phi-2                  |     31.6 |     51.1 |
| mistral-7b             |     35   |     52.3 |
| codellama-7b           |     35.9 |     52.9 |
| wizard-13b             |     36.5 |     51.6 |
| codellama-python-7b    |     37.3 |     57   |
| deepseek-instruct-6.7b |     37.4 |     53.3 |
| mixtral-8x7b           |     39.3 |     59.1 |
| codellama-python-13b   |     39.7 |     56.9 |
| codellama-7b+cot       |     40.4 |     62.8 |
| magicoder-ds-7b        |     41.7 |     62.4 |
| deepseek-base-6.7b     |     41.9 |     62.7 |
| codellama-13b          |     42.5 |     62   |
| wizard-34b             |     42.7 |     57.5 |
| codellama-python-34b   |     43.9 |     59.5 |
| deepseek-base-33b      |     46.5 |     64.9 |
| deepseek-instruct-33b  |     46.5 |     63.2 |
| codellama-34b          |     47.2 |     66.6 |
| phind                  |     47.2 |     63.9 |
| codellama-13b+cot      |     47.4 |     68.4 |
| gpt-3.5-turbo-0613     |     49   |     63.2 |
| codetulu-2-34b         |     49.3 |     68   |
| codellama-34b+cot      |     50.1 |     73.8 |
| gpt-3.5-turbo-0613+cot |     50.3 |     74.9 |
| gpt-4-0613             |     69.8 |     76.8 |
| gpt-4-0613+cot         |     75.5 |     88.9 |

</td><td>

| Model                  |   Pass@1 |   Pass@5 |
|:----------------------:|:--------:|:--------:|
| phi-1                  |     21.7 |     32   |
| phi-1.5                |     27.5 |     39.1 |
| deepseek-instruct-1.3b |     28.7 |     40   |
| codellama-7b+cot       |     29.9 |     55.4 |
| deepseek-base-1.3b     |     31   |     43.4 |
| starcoderbase-7b       |     32.2 |     44.9 |
| phi-2                  |     33.5 |     46.6 |
| codellama-7b           |     34.2 |     48.4 |
| starcoderbase-16b      |     34.2 |     47.1 |
| mistral-7b             |     34.3 |     48.6 |
| codellama-python-7b    |     35.9 |     48.8 |
| codellama-13b+cot      |     36   |     61.8 |
| codellama-13b          |     39.7 |     53.9 |
| phind                  |     39.7 |     52.8 |
| codellama-python-13b   |     39.8 |     52.5 |
| mixtral-8x7b           |     40.5 |     54   |
| deepseek-instruct-6.7b |     41.2 |     52.8 |
| wizard-13b             |     41.3 |     52.4 |
| codellama-python-34b   |     41.4 |     52.9 |
| codellama-34b          |     42.4 |     55.9 |
| wizard-34b             |     43.4 |     53.8 |
| deepseek-base-6.7b     |     43.5 |     54.8 |
| codellama-34b+cot      |     43.6 |     69.4 |
| magicoder-ds-7b        |     44.4 |     57.5 |
| codetulu-2-34b         |     45.8 |     58.9 |
| deepseek-base-33b      |     48.6 |     61.6 |
| gpt-3.5-turbo-0613     |     49.4 |     59.3 |
| deepseek-instruct-33b  |     49.9 |     61.8 |
| gpt-3.5-turbo-0613+cot |     59   |     76.7 |
| gpt-4-0613             |     68.7 |     73   |
| gpt-4-0613+cot         |     77.1 |     88.2 |
</td></tr> </table>
</center>


## 🙏 Acknowledgements
This repository is built on top of [`bigcode-evaluation-harness`](https://github.com/bigcode-project/bigcode-evaluation-harness) and [`FastCode`](https://github.com/Naman-ntc/FastCode), and we thank the contributors of these repos for their awesome works! We also draw inspiration from the [EvalPlus](https://github.com/evalplus/evalplus) repository.

## 📝 Citation
If you find this repository useful, please cite this as
```
TODO
```

## License
CRUXEval is is MIT licensed, as found in the LICENSE file.
