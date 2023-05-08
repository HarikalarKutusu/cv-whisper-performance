# Common Voice - OpenAI/whisper Performance

Measure performance of OpenAI whisper using Common Voice data to create a baseline.

- Whisper is trained using CV v9.0 but how they used it is unknown (probably the whole set).
- So, to get rid of any biasing from measurements, we need to diff (Common Voice latest / v13.0 as of this writing - v9.0) validated, and work with that.
- We get 100 clips from the difference. We chose from longer sentences / thus recordings. We also try to minimize the sentence/speaker biases if there is enough data.
- We feed the audio to multilingual whisper model and get the results, and calculate WER, CER, MER, WIL, WIP values for each recording. Then get some aggregated results for the whole language.
- Repeat it with all supported languages, i.e. intersection of supported by Common Voice & whisper. We also map some locales in Common Voice to Whisper counterparts (currently 4 of them).

## Baseline/Benchmark Process and Results

- The code automatically selected 59 locales, but we excluded one bacause the dataset had no validated records. So we ran code on 58 locales.
- We limited the test set per locale to 100, but we could not reach this limit in all locales as there were not enough data added between v9.0 and v13.0 (3 locales).
- We aimed for unique sentences and unique voices in the test set, but, again due not enough data, we could only reach this goal for 39 locales. As we had an adaptive setting, remaining 19 locales had fewer than 100 distinct voices and/or sentences.
- In total, the test set we produced had 5,622 sentences with 10:38:27 audio.
- We ran this set against all whisper models, including large-v1/v2 distinction.
- We also collected extra data (timing etc), but the overhead was about %0.05, so the main cost is the inference.
- For each model, concurrency is calculated as a function of approximate VRAM required to run the model, as specified in the whisper github repo, with a maximum of 6 real CPU cores on the test setup. Number of processes for 24 GB VRAM has been: 6 for tiny, base and small; 4 for medium, 2 for large-v1&v2 models. We measured timing for model-locale pairs. Because of concurrency, the sum of timings are of course greater than the actual time needed to run the benchmarks.
- On our GPU based configuration, tiny, base and small models had a Real-Time-Factor less than 1 on the average.

## Usage

- Install requirements
- Handle GPU specific case if you intend to use your GPU
- Check `config.py` for paths and other settings and prepare your datasets
- To create a test set, run `python3 compile-delta.py`
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

### Scripts

#### compile-delta.py

Builds a delta dataset to test against

#### compile-baseline.py

Tests the delta dataset against default whisper multi-lingual models.

## Paths to download whisper models

Whisper normally downloads the models into a cache under the user directory, but we liked to free them under the project. You probably have them from prior use, or you can just download them from the links below and put them under data/models/default directory. If you don't have/download them they will be downloaded into data/models/default directory

More info on the default models are [here](https://github.com/openai/whisper#available-models-and-languages).

### Multi-lingual models

- [tiny](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)
- [base](https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt)
- [small](https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt)
- [medium](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt)
- [large-v1](https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt)
- [large-v2](https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt)

### English only models (not used)

[tiny.en](https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt)
[base.en](https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt)
[small.en](https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt)
[medium.en](https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt)

## Test setup

We ran the scripts on a machine with the following specs, thus timing related results are specific for this configuration and will be different on another machine. Therefore, only relative values will be of value.

Software:

- OS: Windows 10x64 22H2 (default balanced power settings)
- PYTHON: Python v3.10.10 using Anaconda venv
- OTHER: We use commonvoice-utils for normalization, jiwer to calculate CER, WER etc values

Hardware:

- CPU: Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz (max 4.34 GHz Turbo) - 6 cores, 12 treads (we used 6 actual cores at max)
- RAM: 48 GB DDR-4
- GPU-0: nVidia GeForce RTX 3090 (dedicated)
- GPU-1: nVidia GeForce GTX 1070 (monitors connected to this graphics card)
- SYSTEM DISK: Samsung nVME SSD 960 EVO 500 GB m2 (temp + v9.0 tsv files)
- WORK DISK: Samsung SSD 860 QVO 1 TB (code & working data)
- DATA DISK: Wester Digital WD100EMAZ 10 TB (initial dataset is already expanded here)
