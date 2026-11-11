# UniVoice Offline (Android)

The voice bot on a phone, with no server and no network: hold the button and
speak, the phone recognizes the speech, picks the scenario, plays the
pre-recorded answer and keeps the conversation as a chat with both
recordings. The network is needed once, to download the speech models (or
not at all if the model files are copied to the phone).

| Part | How it works on the phone |
|---|---|
| Speech recognition (default) | [Vosk](https://alphacephei.com/vosk/) small models, ~50 MB per language (Uzbek 51 MB, Russian 46 MB zip), several times faster than real time even on old phones. On 99 synthesized scenario phrases it reaches the right scenario as often as the server's Whisper large-v3 (96 vs 95, `docs/stt-evaluation.md`). |
| Speech recognition (optional) | Modellar → Whisper: [whisper.cpp](https://github.com/ggml-org/whisper.cpp) through JNI. Uzbek: the project's fine-tune ([ZOKHID/univoice-stt-uz](https://huggingface.co/ZOKHID/univoice-stt-uz)) converted to ggml q5_0 (~0.5 GB). Russian: Whisper small q5_1 (190 MB), because the Uzbek model has forgotten Russian. The language is picked with the UZ / RU switch. |
| Scenario | A Kotlin port of `app/services/scenario_service.py`; `scenarios.json` and `prompts.json` are packed from `app/data` at build time, so the phone and the call bot answer alike. |
| Answer | WAV files rendered ahead of time by `scripts/export_mobile_prompts.py` into `app/src/main/assets/prompts`. |
| Chat | SQLite on the phone: each caller turn keeps its recording, transcript, language and recognition time; each bot turn its scenario, confidence and answer. History, replay, and an "export recordings" zip for `scripts/eval_stt.py`. |
| Speed test | Menu → Tezlik testi / Тест скорости: recognition time on a bundled 6 s clip and the latest recordings. |

Two ways to use it: hold the button and talk to the bot (chat), or switch on
the **answering machine** and let it pick up the phone (below).

Not in this version: live speech synthesis, iOS.

## Answering machine

The "Avtojavob · Автоответчик" switch under the title bar starts a
foreground service (with a notification that has an off button). While it
runs, the phone answers incoming calls by itself, offline:

1. After a few rings (Settings: 0–10 s) it picks up and puts the call on speaker.
2. It greets the caller in both languages, then listens with both Vosk models
   at once (Uzbek and Russian); the more confident scenario wins, and the
   answer comes in that language. On the 99 test phrases with silence around
   them this finds the right scenario for 92 (93%).
3. It plays the pre-recorded answer and asks "anything else?". Silence after
   an answer ends the call with a goodbye; silence before one asks again
   once, then says goodbye. The same timers as the server
   (`REALTIME_NO_INPUT_SECONDS`, `REALTIME_FOLLOWUP_SECONDS`); the longest
   call is a setting (1–10 min).
4. A phone cannot transfer a call, so a request for an operator, or two
   replies it did not understand, end with "your request is recorded, an
   operator will call you back". That call is marked ↩ in the history, and
   its chat shows a button that dials the caller back.
5. Every call is a conversation in the history (caller's number, both sides'
   words, the caller's recordings for the export), shown live if the app is open.

Needs Android 9 or newer (earlier versions cannot hang up), the Vosk models
(Whisper cannot stream), and these permissions, asked when the switch is
turned on: microphone, phone, answer calls, notifications, and the call log
(only to learn the caller's number; it can be refused). On Samsung phones also
take the app out of "sleeping apps" so Android does not stop the service.

Limits to know:

- Android gives an app no way to play sound into a call or to hear the
  caller's line. The phone is on speaker: the answers come out of its speaker
  into its microphone, and the caller's voice comes out of the speaker into
  the microphone. The speaker volume is set to maximum for the call.
- Since Android 10 a phone feeds silence to every app that records during a
  call, unless the app has an accessibility service switched on. The app
  ships one that does nothing else ("UniVoice Avtojavob": Settings ->
  Accessibility -> Downloaded apps); the chat shows a button to it while it
  is off. On Android 13+ a sideloaded app first needs App info -> menu ->
  "Allow restricted settings", and Play Protect may refuse to install it
  from a file until its app scanning is paused.
- Tested phones: Redmi Note 12 (Android 15) hears the caller with the
  service on. Galaxy S8 (Android 9) gives apps no microphone data at all in
  a call (the audio driver returns nothing, every source), so it cannot work
  there; the chat says "Mikrofon jim" (microphone is mute) and the bot only
  greets. Whether the caller hears the bot depends on the phone's echo
  cancelling; check it with a real call.
- With a USB sound card attached the app records from and plays through it
  instead (its input and output found automatically), for wiring the call
  through the headset jack by cable.
- The bot does not hear the caller while it talks (no interrupting).
- The service stops when the phone restarts: switch it on again.
- The callers' voices are recorded and kept on the phone; many countries
  require telling callers. Put that in the greeting (`app/data/prompts.json`,
  then `python -m scripts.export_mobile_prompts`).

In a debug build the call can be driven without a caller or a microphone:
push a WAV to `/sdcard/Android/data/uz.univoice.offline/files/call_input.wav`,
then simulate the call in the emulator with `adb emu gsm call 998901234567`;
the bot hears the file as the caller's first sentence.

## Build

Needs the Android SDK with NDK `29.0.14206865` and CMake `4.1.2`
(`sdkmanager "ndk;29.0.14206865" "cmake;4.1.2"`), JDK 17.

1. whisper.cpp is a git submodule, pinned to v1.9.5:

       git submodule add https://github.com/ggml-org/whisper.cpp.git mobile/android/third_party/whisper.cpp
       git -C mobile/android/third_party/whisper.cpp checkout v1.9.5

   (on a fresh clone of this repository: `git submodule update --init`).
   Without it the app still builds and runs with Vosk, the default engine;
   only the Whisper engine is missing.

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

   `-Pemulator` adds an x86_64 library for the Android emulator: its ARM
   translation crashes inside whisper.cpp (SIGILL). Recognition on the
   emulator only checks that it works; measure speed on real phones.

## Uzbek model

whisper.cpp needs the fine-tune in its own format. From the repository root:

    pip install transformers torch numpy
    cmake -S mobile/android/third_party/whisper.cpp -B build/whisper -DCMAKE_BUILD_TYPE=Release
    cmake --build build/whisper --target whisper-quantize -j
    python -m scripts.convert_whisper_ggml --quantize-bin build/whisper/bin/whisper-quantize

This writes `models/ggml/ggml-univoice-stt-uz-q5_0.bin` (539 212 484 bytes)
and prints its size and sha256; `Models.UZ` holds them, so the app checks the
download. The app downloads it from `ZOKHID/univoice-stt-uz-ggml` on Hugging
Face; or copy it to the phone and use Modellar → Fayldan.

On Windows without a C compiler, run the same commands in a
`python:3.12-slim` container (`apt-get install cmake build-essential` first).
On the emulator the converted model transcribes the bundled clip as "internet
ishlamayapti modemda qizil chiroq yonib turibdi".

The app runs it the way the server does (`docs/stt-evaluation.md`): Uzbek
forced, greedy decoding, no timestamps (they made it drop the first words),
no prompt (it continued the prompt instead of transcribing), no temperature
fallback.

## Install and first run

    adb install app/build/outputs/apk/debug/app-debug.apk

(or copy the APK to the phone and open it; allow installing from this source).

1. Menu → Modellar / Модели: Vosk is chosen; download the Uzbek and
   Russian models (~100 MB together) or import their .zip files
   (`vosk-model-small-uz-0.22.zip` and `vosk-model-small-ru-0.22.zip`
   from alphacephei.com, mirrored on Hugging Face as `ZOKHID/univoice-vosk`;
   checked by sha256 and unpacked by the app).
   Whisper is there for phones fast enough for it.
2. Menu → Sozlamalar / Настройки: the operator's phone number for the
   "call an operator" button shown when the bot did not understand.
3. Menu → Tezlik testi: run it on each phone you plan to use. "×" is
   recognition time divided by audio length; above ~1.5 the answer arrives
   too late to feel like a conversation.
4. Hold the microphone button, speak, release.

Vosk needs about 300 MB of memory. The Whisper Uzbek model needs about
1.5 GB of free memory.

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
