# Common Voice - OpenAI/whisper Performance

Measure performance of OpenAI whisper using Common Voice data.

- Whisper is trained using CV v9.0 but how they used it is unknown (probably the whole set).
- So, to get rid of any biasing from measurements, we need to diff (v13.0 - v9.0) validated, and work with that.
- We get a pseudo-random 1000 clips from the difference. I said pseudo, because it might be better to choose from longer sentences / thus recordings.
- We feed the audio to multilingual whisper model and get the results, and calculate a similarity distance (to be specified), for each recording. Then get some aggregated results for the whole language.
- Repeat it with all supported languages (intersection of supported by omnilingo & whisper - didn't check the lists yet).

## Paths to download whisper models

To programmatically use whisper, you need to download the models you want to test against and put them under data/models directory. More info on the default models are [here](https://github.com/openai/whisper#available-models-and-languages).

### Multi-lingual models

[tiny](https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt)
[base](https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt)
[small](https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt)
[medium](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt)
[large](https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large.pt)

### English only models

[tiny.en](https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt)
[base.en](https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt)
[small.en](https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt)
[medium.en](https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt)
