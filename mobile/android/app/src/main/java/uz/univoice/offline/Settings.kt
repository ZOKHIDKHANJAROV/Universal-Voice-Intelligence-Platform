package uz.univoice.offline

import android.content.Context
import uz.univoice.offline.stt.Engine
import uz.univoice.offline.stt.ModelSpec

/** The few things a user can change, kept in SharedPreferences. */
class Settings(context: Context) {
    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)

    var language: String
        get() = prefs.getString("language", "uz") ?: "uz"
        set(value) = prefs.edit().putString("language", value).apply()

    /** Number the "call an operator" button dials. Empty hides the button's number. */
    var operatorPhone: String
        get() = prefs.getString("operator_phone", "") ?: ""
        set(value) = prefs.edit().putString("operator_phone", value.trim()).apply()

    /** Vosk by default: a 50 MB model that runs in real time even on old phones. */
    var engine: Engine
        get() = runCatching { Engine.valueOf(prefs.getString("engine", null) ?: "VOSK") }.getOrDefault(Engine.VOSK)
        set(value) = prefs.edit().putString("engine", value.name).apply()

    /** whisper.cpp threads; 4 suits the big cores of most phones. */
    var threads: Int
        get() = prefs.getInt("threads", defaultThreads())
        set(value) = prefs.edit().putInt("threads", value.coerceIn(1, 8)).apply()

    fun modelUrl(spec: ModelSpec): String = prefs.getString("model_url_${spec.id}", null) ?: spec.defaultUrl

    fun setModelUrl(spec: ModelSpec, url: String) {
        val trimmed = url.trim()
        prefs.edit().apply {
            if (trimmed.isEmpty() || trimmed == spec.defaultUrl) remove("model_url_${spec.id}")
            else putString("model_url_${spec.id}", trimmed)
        }.apply()
    }

    private fun defaultThreads() = Runtime.getRuntime().availableProcessors().coerceIn(2, 4)
}
