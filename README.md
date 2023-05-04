# Common Voice - OpenAI/whisper Performance

Measure performance of OpenAI whisper using Common Voice data.

- Whisper is trained using CV v9.0 but how they used it is unknown (probably the whole set).
- So, to get rid of any biasing from measurements, we need to diff (v13.0 - v9.0) validated, and work with that.
- We get a pseudo-random 1000 clips from the difference. I said pseudo, because it might be better to choose from longer sentences / thus recordings.
- We feed the audio to multilingual whisper model and get the results, and calculate a similarity distance (to be specified), for each recording. Then get some aggregated results for the whole language.
- Repeat it with all supported languages (intersection of supported by omnilingo & whisper - didn't check the lists yet).
