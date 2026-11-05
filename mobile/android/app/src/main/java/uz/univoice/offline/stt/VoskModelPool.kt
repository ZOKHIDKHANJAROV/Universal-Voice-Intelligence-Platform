package uz.univoice.offline.stt

import org.vosk.Model

/**
 * Loaded Vosk models, shared by the chat screen and the call answering
 * service so a model is in memory once. Reference counted: the last
 * [release] frees it.
 */
object VoskModelPool {
    private class Entry(val model: Model) { var refs = 0 }

    private val entries = HashMap<String, Entry>()

    @Synchronized
    fun acquire(path: String): Model {
        val entry = entries.getOrPut(path) { Entry(Model(path)) }
        entry.refs++
        return entry.model
    }

    @Synchronized
    fun release(path: String) {
        val entry = entries[path] ?: return
        if (--entry.refs <= 0) {
            entries.remove(path)
            entry.model.close()
        }
    }
}
