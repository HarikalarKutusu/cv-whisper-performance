# Common Voice - OpenAI/whisper Performance

Measure performance of OpenAI whisper using Common Voice data to create a baseline.

- Whisper is trained using CV v9.0 but how they used it is unknown (probably the whole set).
- So, to get rid of any biasing from measurements, we need to diff (Common Voice latest (v13.0 as of this writing) - v9.0) validated, and work with that.
- We get 100 clips from the difference. We chose from longer sentences / thus recordings. We also try to minimize the sentence/speaker biases if there are enough data.
- We feed the audio to multilingual whisper model and get the results, and calculate a similarity distance (to be specified), for each recording. Then get some aggregated results for the whole language.
- Repeat it with all supported languages, i.e. intersection of supported by Common Voice & whisper. We also map some detailed locales in Common Voice to Whisper counterparts.

## Usage

## Install

### Configuration

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
