package uz.univoice.offline.stt

import android.util.Log
import uz.univoice.offline.BuildConfig
import java.io.Closeable

/**
 * whisper.cpp through JNI (src/main/cpp/whisper_jni.cpp). One instance holds
 * one loaded model; it is not thread-safe, so the caller serializes access.
 */
class WhisperEngine private constructor(private var handle: Long) : Closeable {

    /**
     * Transcribes 16 kHz mono audio in [language] ("uz", "ru"). Runs the way
     * the server runs the Uzbek model: greedy, no timestamps, no prompt.
     */
    fun transcribe(audio: FloatArray, language: String, threads: Int): String {
        check(handle != 0L) { "model is closed" }
        return String(nativeTranscribe(handle, audio, language, threads), Charsets.UTF_8).trim()
    }

    override fun close() {
        if (handle != 0L) {
            nativeFree(handle)
            handle = 0L
        }
    }

    companion object {
        private const val TAG = "WhisperEngine"

        /** False when the app was built without third_party/whisper.cpp. */
        val available: Boolean by lazy {
            BuildConfig.WITH_WHISPER && runCatching { System.loadLibrary("univoice_whisper") }
                .onFailure { Log.e(TAG, "native library failed to load", it) }
                .isSuccess
        }

        fun systemInfo(): String = if (available) nativeSystemInfo() else "whisper.cpp not built in"

        fun load(modelPath: String): WhisperEngine {
            check(available) { "speech recognition is not built into this app" }
            val handle = nativeInit(modelPath)
            check(handle != 0L) { "could not load model $modelPath" }
            return WhisperEngine(handle)
        }

        @JvmStatic private external fun nativeInit(modelPath: String): Long
        @JvmStatic private external fun nativeFree(handle: Long)
        @JvmStatic private external fun nativeTranscribe(handle: Long, audio: FloatArray, language: String, threads: Int): ByteArray
        @JvmStatic private external fun nativeSystemInfo(): String
    }
}
