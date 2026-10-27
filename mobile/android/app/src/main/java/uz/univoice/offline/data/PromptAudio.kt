package uz.univoice.offline.data

import android.content.res.AssetManager
import org.json.JSONObject

/**
 * Pre-recorded answers bundled in assets/prompts, listed in manifest.json by
 * scripts/export_mobile_prompts.py: a scenario id or a prompts.json key, per
 * language, to a WAV file.
 */
class PromptAudio(private val assets: AssetManager) {
    private data class Entry(val file: String, val text: String)

    private val entries: Map<Pair<String, String>, Entry> = runCatching {
        val root = JSONObject(assets.open("prompts/manifest.json").bufferedReader().use { it.readText() })
        val list = root.getJSONArray("entries")
        (0 until list.length()).associate { i ->
            val item = list.getJSONObject(i)
            (item.getString("key") to item.getString("language")) to
                Entry("prompts/" + item.getString("file"), item.getString("text"))
        }
    }.getOrDefault(emptyMap())

    val count: Int get() = entries.size

    /**
     * Asset path of the answer, or null when it was never rendered or its text
     * changed since (the recording would no longer match what is shown).
     */
    fun assetFor(key: String, language: String, text: String): String? =
        entries[key to language]?.takeIf { it.text == text }?.file
}
