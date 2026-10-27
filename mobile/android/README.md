# UniVoice Offline (Android)

The voice bot on a phone, with no server and no network: hold the button and
speak, the phone recognizes the speech, picks the scenario, plays the
pre-recorded answer and keeps the conversation as a chat with both
recordings. The network is needed once, to download the speech models (or
not at all if the model files are copied to the phone).

| Part | How it works on the phone |
|---|---|
| Speech recognition | [whisper.cpp](https://github.com/ggml-org/whisper.cpp) through JNI. Uzbek: the project's fine-tune ([ZOKHID/univoice-stt-uz](https://huggingface.co/ZOKHID/univoice-stt-uz)) converted to ggml q5_0 (~0.5 GB). Russian: Whisper small q5_1 (190 MB), because the Uzbek model has forgotten Russian. The language is picked with the UZ / RU switch. |
| Scenario | A Kotlin port of `app/services/scenario_service.py`; `scenarios.json` and `prompts.json` are packed from `app/data` at build time, so the phone and the call bot answer alike. |
| Answer | WAV files rendered ahead of time by `scripts/export_mobile_prompts.py` into `app/src/main/assets/prompts`. |
| Chat | SQLite on the phone: each caller turn keeps its recording, transcript, language and recognition time; each bot turn its scenario, confidence and answer. History, replay, and an "export recordings" zip for `scripts/eval_stt.py`. |
| Speed test | Menu → Tezlik testi / Тест скорости: recognition time on a bundled 6 s clip and the latest recordings. |

Not in this version: automatic end-of-speech detection (push-to-talk
instead), automatic language detection, live speech synthesis, iOS.

## Build

Needs the Android SDK with NDK `29.0.14206865` and CMake `4.1.2`
(`sdkmanager "ndk;29.0.14206865" "cmake;4.1.2"`), JDK 17.

1. whisper.cpp is a git submodule, pinned to v1.9.5:

       git submodule add https://github.com/ggml-org/whisper.cpp.git mobile/android/third_party/whisper.cpp
       git -C mobile/android/third_party/whisper.cpp checkout v1.9.5

   (on a fresh clone of this repository: `git submodule update --init`).
   Without it the app still builds and runs — chat, scenarios, answers — but
   shows "no speech recognition in this build".

2. Answers (from the repository root; TTS services running as for
   `make render-prompts`, or with `--from-cache` to reuse the 8 kHz call audio):

       python -m scripts.export_mobile_prompts

3. The APK:

       cd mobile/android
       ./gradlew assembleDebug          # app/build/outputs/apk/debug/app-debug.apk
       ./gradlew testDebugUnitTest      # includes the scenario parity test

   `-ParmArch=armv8.2-a+dotprod` builds faster recognition for phones with
   Cortex-A55/A75 or newer cores; that build crashes on older (armv8.0)
   phones, so measure with the default build first.

## Uzbek model

whisper.cpp needs the fine-tune in its own format. From the repository root:

    pip install transformers torch numpy
    cmake -S mobile/android/third_party/whisper.cpp -B build/whisper -DCMAKE_BUILD_TYPE=Release
    cmake --build build/whisper --target whisper-quantize -j
    python -m scripts.convert_whisper_ggml --quantize-bin build/whisper/bin/whisper-quantize

This writes `models/ggml/ggml-univoice-stt-uz-q5_0.bin` and prints its size
and sha256. Either publish it (the app's default address is
`ZOKHID/univoice-stt-uz-ggml` on Hugging Face; set `sizeBytes` and `sha256`
in `Models.UZ`) or copy it to the phone and use Modellar → Fayldan.

The app runs it the way the server does (`docs/stt-evaluation.md`): Uzbek
forced, greedy decoding, no timestamps (they made it drop the first words),
no prompt (it continued the prompt instead of transcribing), no temperature
fallback.

## Install and first run

    adb install app/build/outputs/apk/debug/app-debug.apk

(or copy the APK to the phone and open it; allow installing from this source).

1. Menu → Modellar / Модели: download or import the Uzbek and Russian models.
2. Menu → Sozlamalar / Настройки: the operator's phone number for the
   "call an operator" button shown when the bot did not understand.
3. Menu → Tezlik testi: run it on each phone you plan to use. "×" is
   recognition time divided by audio length; above ~1.5 the answer arrives
   too late to feel like a conversation.
4. Hold the microphone button, speak, release.

The model needs a 64-bit phone with about 1.5 GB of free memory.

## Exported recordings

Menu → Yozuvlarni eksport / Экспорт записей shares a zip:
`audio/*.wav` (16 kHz) and `manifest.csv` with the columns
`scripts/eval_stt.py` reads. `text` and `scenario` start as what the phone
recognized and chose; correct them by hand, then

    python -m scripts.eval_stt path/to/manifest.csv --language uz

Callers must be told that they are recorded.

## Keeping the phone and the server in step

The scenario matcher is ported, not shared. After changing
`app/services/scenario_service.py` or `app/data/scenarios.json`:

    python -m scripts.export_intent_golden   # from the repository root
    cd mobile/android && ./gradlew testDebugUnitTest

`ScenarioParityTest` replays every phrase in the golden file and fails on any
difference in scenario or confidence.
