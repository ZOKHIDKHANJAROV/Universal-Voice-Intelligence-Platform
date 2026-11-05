package uz.univoice.offline.stt

import android.util.Log
import org.json.JSONObject
import org.vosk.LibVosk
import org.vosk.LogLevel
import org.vosk.Model
import org.vosk.Recognizer

/**
 * A small Vosk model with its whole vocabulary. On the 99 synthesized
 * scenario phrases it picks the right scenario as often as the server's
 * Whisper large-v3 (scripts/eval_keyword_stt.py), several times faster than
 * real time on a phone. Limiting it to the scenario words was worse: it forces
 * every sound into a keyword and invents complaints.
 */
class VoskEngine private constructor(private val path: String, private val model: Model) : SpeechEngine {
    private var closed = false

    override fun transcribe(audio: FloatArray, language: String, threads: Int): String {
        val pcm = ShortArray(audio.size) { (audio[it] * 32768f).coerceIn(-32768f, 32767f).toInt().toShort() }
        return newStream().use { stream ->
            stream.accept(pcm)
            stream.finish()
        }
    }

    override fun startStream(language: String): SpeechStream = newStream()

    /**
     * [phraseEndSeconds]: how long a pause ends a phrase; null keeps the
     * model's own (about half a second). Several streams may run at once on
     * one engine (the call answerer listens in two languages).
     */
    fun newStream(phraseEndSeconds: Float? = null): SpeechStream {
        val recognizer = Recognizer(model, 16000f)
        // Initial silence up to 30 s and phrases up to 15 s are not cut short.
        if (phraseEndSeconds != null) recognizer.setEndpointerDelays(30f, phraseEndSeconds, 15f)
        return VoskStream(recognizer)
    }

    @Synchronized
    override fun close() {
        if (closed) return
        closed = true
        VoskModelPool.release(path)
    }

    private class VoskStream(private val recognizer: Recognizer) : SpeechStream {
        // A pause inside a sentence ends a phrase for Vosk; keep every phrase.
        private val phrases = mutableListOf<String>()

        override fun accept(audio: ShortArray): Boolean {
            val ended = recognizer.acceptWaveForm(audio, audio.size)
            if (ended) add(recognizer.result, "text")
            return ended
        }

        override fun partial(): String = words(JSONObject(recognizer.partialResult).optString("partial"))

        override fun take(): String = phrases.joinToString(" ").also { phrases.clear() }

        override fun finish(): String {
            add(recognizer.finalResult, "text")
            return take()
        }

        override fun close() = recognizer.close()

        private fun add(json: String, key: String) {
            val text = words(JSONObject(json).optString(key))
            if (text.isNotEmpty()) phrases += text
        }

        private fun words(text: String) =
            text.split(' ').filter { it.isNotBlank() && it != "[unk]" }.joinToString(" ")
    }

    companion object {
        private const val TAG = "VoskEngine"

        val available: Boolean by lazy {
            runCatching { LibVosk.setLogLevel(LogLevel.WARNINGS) }
                .onFailure { Log.e(TAG, "native library failed to load", it) }
                .isSuccess
        }

        fun load(modelDir: String): VoskEngine {
            check(available) { "Vosk is not built into this app" }
            return VoskEngine(modelDir, VoskModelPool.acquire(modelDir))
        }
    }
}
