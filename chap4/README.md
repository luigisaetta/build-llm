# Chapter 4: load the pretrained GPT-2 weights

This chapter loads the public weights of **GPT-2 Small** into the `GPTModel`
of [Chapter 3](../chap3/README.md), and uses it to continue a text. With
random weights the model writes nonsense; with the GPT-2 weights, the same
architecture writes fluent English.

The book loads the original OpenAI weights, which are TensorFlow checkpoints,
and reads them with TensorFlow. This repository does not use TensorFlow: it
downloads the same weights from Hugging Face, in the
[safetensors](https://huggingface.co/docs/safetensors) format, and reads them
with the small `safetensors` library. The notebook explains the differences
and the conversion step by step.

## Notebooks

1. **[Load the GPT-2 Small weights and generate text](load_gpt2_weights.ipynb)**:
   download `model.safetensors` from
   [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2),
   inspect the tensors of the checkpoint, and map them to the parameters of
   `GPTModel`: rename them, split the fused query, key, and value projection,
   transpose the `Conv1D` weights, and copy the token embeddings into the
   output head (GPT-2 uses weight tying). Then load the weights, tokenize a
   prompt, and generate a continuation with greedy decoding.

The reusable code is in the `build_llm` package:

* [`build_llm/gpt2.py`](../build_llm/gpt2.py): `GPTModel` and
  `generate_text_simple`;
* [`build_llm/gpt2_weights.py`](../build_llm/gpt2_weights.py): the GPT-2
  Small configuration, the download with checksum verification, and the
  conversion of the weights.

## Prerequisites

* The `build-llm` Conda environment, with the dependencies and the editable
  installation of the [Quickstart](../QUICKSTART.md). The `safetensors`
  package is in `requirements.txt`.
* Network access for the first run, to download about 548 MB from Hugging
  Face, and to let tiktoken download the GPT-2 encoding if it is not cached.
* About 2 GB of free memory: the checkpoint and the model are both in memory
  while the weights are loaded.

## Run the notebook

From the repository root:

```bash
conda activate build-llm
```

```bash
python -m jupyterlab
```

Open `load_gpt2_weights.ipynb` with the kernel of the `build-llm` environment,
and run it from top to bottom. The model runs on the Apple Silicon GPU (MPS)
when available, otherwise on CUDA or the CPU.

Expected result: the prompt "Every effort moves you" continues with
"forward. The first step is to understand the importance of your work.",
and a second prompt shows the repetitions typical of greedy decoding.

## Downloaded files and cleanup

The weights are saved in `models/gpt2-small/model.safetensors`, in the
repository root. The `models/` folder is ignored by Git. The download uses a
fixed revision of the Hugging Face repository, and the file is accepted only
if its SHA-256 checksum matches the published one. Later runs reuse the file.

To free the disk space, delete the folder:

```bash
rm -r models/gpt2-small
```

## Troubleshooting

* **`ModuleNotFoundError: safetensors`**: install the dependencies again with
  `pip install -r requirements.txt`, then restart the kernel.
* **`Checksum mismatch`**: the download was corrupted, and the partial file
  was removed. Run the cell again.
* **Download errors**: check the network connection, and that
  `huggingface.co` is reachable.

## Scope and next steps

The notebook uses greedy decoding, which always picks the most likely token:
the output is deterministic but tends to repeat itself. The next steps are
temperature and top-k sampling, and then the training loop.

## Development checks

From the repository root, with the `build-llm` environment active:

```bash
black .
```

```bash
pylint build_llm chap* tests
```

```bash
pytest
```

The tests of `build_llm/gpt2_weights.py` run without network access: they use
small random checkpoints and a fake download. A network test checks that
Hugging Face still publishes the expected file, without downloading it:

```bash
pytest -m network
```

The test suite does not execute the notebook. Verify it separately with a
fresh kernel and **Run All Cells**.
