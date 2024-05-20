# Common Voice - OpenAI/whisper Performance

Measure performance of OpenAI whisper using Common Voice data to create a baseline.

Whisper is not trained using Common Voice data, but -initially- evaluated using CV v9.0. In October 2023, about one year after the initial release, with the large-v3 model release, OpenAI tested the large-v3 model against Common Voice v15.0 dataset.

Whisper multilingual performance is usually bad for low-resourced languages. So, using the latest Common Voice data, we can fine-tune the whisper models for better inference.

But before doing this, we need a baseline to compare it against, and thus the reason of this repository. We also want some performance measurements and timing/durations to compare inference on different, possibly low end environments, such as a web prowser implementation.

During this work we wanted to relax the randomness of the test to be able to get best possible results:

- We normalized the data using commonvoice-utils (where possible), so CER/WER errors caused by punctuation/character case does not effect the measures.
- We know, for SotA models such as Whisper, longer recordings (thus sentences) are better, usually in the 5-25 sec range. On the other hand, Common Voice limits the recordings with 10 sec and the average is about among 4-6 sec for many languages. To get better performance, we try to select "longest 100 sentences" from each language as a test set.
- To also minimize any voice/transcription bias, we

## Algorithm/Steps

1. Create a test set (`compile-test-set.py`).
   - Use latest Common Voice release and use only locales which are common to Whisper and Common Voice
   - Use the validated.tsv file for each locale and normalize the sentences, sort by longest to shortest
   - Try to minimize the sentence/speaker biases by selecting single speaker, single sentence (if there is enough data).
   - Select N longest clips (default 100) to create the test-set
2. Run inference tests against the default models (`compile-baseline.py`)
   - For each locale, run whisper-translate, calculate durations and metrics (WER, CER, MER, WIL, WIP, and RTF/Real-Time-Factor)
   - Aggeregate the results

AFter we have this baseline (not in this repo):

1. Fine tune for different languages
2. Run inference tests in this repo's test sets againt the new fine-tuned models
3. Compare the results

## Baseline/Benchmark Process and Results

We first run the initial versions of this code against CV v13.0, then for some select languages against v14.0 (not included here). Now, while making this repo public, we give results for Common Voice v15.0 also.

- The code automatically selected 59 locales, but we excluded one because the dataset had no validated records. So we ran code on 58 locales.
- We limited the test set per locale to 100, only `az` locale had only 87 recordings.
- We aimed for unique sentences and unique voices in the test set, but, again due not enough data, we could only reach this goal for 46 locales. As we had an adaptive setting, remaining 12 locales had fewer than 100 distinct voices and/or sentences.
- In total, the test set we produced had 5,787 sentences/recordings with 11:49:44 audio.
- We ran this set against all whisper models, including large-v1/v2 distinction.
- We also collected extra data (timing etc), but the overhead was about %0.05, so the main cost is the inference.
- For each model, concurrency is calculated as a function of approximate VRAM required to run the model, as specified in the whisper github repo, with a maximum of 6 real CPU cores on the test setup. Number of processes for 24 GB VRAM has been: 6 for tiny, base and small; 4 for medium, 2 for large-v1&v2 models. We measured timing for model-locale pairs. Because of concurrency, the sum of timings are of course greater than the actual time needed to run the benchmarks.
- On our GPU based configuration, tiny, base and small models had a Real-Time-Factor less than 1 on the average.

## Usage

- Install requirements
- Handle GPU specific case if you intend to use your GPU
- Check `config.py` for paths and other settings and prepare your datasets
- To create a test set, run `python3 compile-test-set.py`
- To test it, run `python3 compile-baseline.py`

TODO:

- Add fine-tuning scripts and expand existing code to test them

## Install

```sh
pip install -U -r requirements.txt
```

## GPU Use

You do NOT need to install CUDA Toolkit etc to your system, because the following package already includes them.

```sh
# uninstall CPU only versio and install GPU only version
pip uninstall torch
pip cache purge
pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# if you want to monitor nVidia card from pytorch
pip3 install -U pynvml
```

### Configuration

The configuration data is in the config.py file. Please read the comments there for their usage.

## Paths to download whisper models

Whisper normally downloads the models into a cache under the user directory, but we already keep them in our system. You probably have them from prior use, or you can just download them from the links below and put them on a large drive and point to them from `config.py`.

More info on the default models are [here](https://github.com/openai/whisper#available-models-and-languages).

### Multi-lingual models

- [tiny](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)
- [base](https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt)
- [small](https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt)
- [medium](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt)
- [large-v1](https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt)
- [large-v2](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt)
- [large-v3](https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt)

### English only models (not used)

[tiny.en](https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt)
[base.en](https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt)
[small.en](https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt)
[medium.en](https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt)

## Test setup

We ran the scripts on a machine with the following specs, thus timing related results are specific to this configuration and will be different on another machine. Therefore, only relative values will be of value.

Software:

- OS:
  - Until v15.0: Windows 10x64 22H2 (default balanced power settings)
  - With v16.1: Windows 11x64 23H2 (default balanced power settings)
- PYTHON: Python v3.11.6 using Anaconda venv
- OTHER: We use commonvoice-utils for normalization (where supported), jiwer to calculate CER, WER etc values

Hardware:

- CPU: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz (max 4.3 GHz Turbo) - 6 cores, 12 treads (we used 6 actual cores at max on GPU based inference and max 2 cores for CPU based inference)
- RAM: 48 GB DDR-4 @ 3 Ghz XMP
- GPU-0: nVidia GeForce RTX 3090 (dedicated for inference)
- GPU-1: nVidia GeForce GTX 1070 (monitors connected to this graphics card)
- DISKS UNTIL v15.0:
  - SYSTEM DISK: Samsung nVME SSD 960 EVO 500 GB m2 (temp directory)
  - LOCAL REPO DISK: Samsung SSD 860 QVO 1 TB (code & working data)
  - DATA DISK: Western Digital WD100EMAZ 10 TB (models + initial dataset is already expanded here)
- DISKS STARTING WITH v16.1:
  - SYSTEM & REPO DISK: Samsung nVME SSD 990 Pro 2 TB m2 (inc. temp directory)
  - DATA DISK: Western Digital WD100EMAZ 10 TB (models + initial dataset is already expanded here)
