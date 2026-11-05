package uz.univoice.offline.stt

import java.io.Closeable

/** One loaded speech model. Not thread-safe: the caller serializes access. */
interface SpeechEngine : Closeable {
    /** Transcribes 16 kHz mono audio (floats in -1..1) spoken in [language]. */
    fun transcribe(audio: FloatArray, language: String, threads: Int): String
}

enum class Engine(val title: String) {
    /** Kaldi small models: ~50 MB per language, real time on any phone. */
    VOSK("Vosk · yengil / лёгкий"),

    /** whisper.cpp: 190-540 MB per language, needs a fast phone. */
    WHISPER("Whisper · katta / большой"),
}
