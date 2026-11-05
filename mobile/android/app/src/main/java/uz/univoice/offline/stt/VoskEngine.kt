package uz.univoice.offline.stt

import android.util.Log
import org.json.JSONObject
import org.vosk.LibVosk
import org.vosk.LogLevel
import org.vosk.Model
import org.vosk.Recognizer

/**
 * A small Vosk (Kaldi) model with its whole vocabulary. On the 99 synthesized
 * scenario phrases it picks the right scenario as often as the server's
 * Whisper large-v3 (scripts/eval_keyword_stt.py), several times faster than
 * real time on a phone. Limiting it to the scenario words was worse: it forces
 * every sound into a keyword and invents complaints.
 */
class VoskEngine private constructor(private val model: Model) : SpeechEngine {

    override fun transcribe(audio: FloatArray, language: String, threads: Int): String {
        val pcm = ShortArray(audio.size) { (audio[it] * 32768f).coerceIn(-32768f, 32767f).toInt().toShort() }
        return startStream(language).use { stream ->
            stream.accept(pcm)
            stream.finish()
        }
    }

    override fun startStream(language: String): SpeechStream = object : SpeechStream {
        private val recognizer = Recognizer(model, 16000f)

        override fun accept(audio: ShortArray) {
            recognizer.acceptWaveForm(audio, audio.size)
        }

        override fun finish(): String = JSONObject(recognizer.finalResult).optString("text")
            .split(' ').filter { it.isNotBlank() && it != "[unk]" }.joinToString(" ")

        override fun close() = recognizer.close()
    }

    override fun close() = model.close()

    companion object {
        private const val TAG = "VoskEngine"

        val available: Boolean by lazy {
            runCatching { LibVosk.setLogLevel(LogLevel.WARNINGS) }
                .onFailure { Log.e(TAG, "native library failed to load", it) }
                .isSuccess
        }

        fun load(modelDir: String): VoskEngine {
            check(available) { "Vosk is not built into this app" }
            return VoskEngine(Model(modelDir))
        }
    }
}
