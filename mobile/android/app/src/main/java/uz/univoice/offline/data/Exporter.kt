package uz.univoice.offline.data

import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Packs every recorded caller turn into a zip for scripts/eval_stt.py.
 *
 * manifest.csv has the columns eval_stt.py reads (audio, text, language,
 * scenario). text and scenario start as what the phone recognized and chose:
 * correct them by hand to turn the export into a reference set.
 */
class Exporter(private val store: ChatStore) {

    fun export(target: File): Int {
        val turns = store.callerRecordings().reversed().filter { File(it.audioPath!!).exists() }
        target.parentFile?.mkdirs()
        ZipOutputStream(target.outputStream().buffered()).use { zip ->
            val manifest = StringBuilder()
            manifest.append("audio,text,language,scenario,stt_text,bot_scenario,bot_answer,stt_ms,conversation,recorded_at\n")
            val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.ROOT)
            for (turn in turns) {
                val audio = File(turn.audioPath!!)
                val name = "audio/${audio.name}"
                zip.putNextEntry(ZipEntry(name))
                audio.inputStream().use { it.copyTo(zip) }
                zip.closeEntry()
                val reply = store.replyTo(turn)
                manifest.append(
                    listOf(
                        name, turn.text, turn.language.orEmpty(), reply?.scenarioId.orEmpty(),
                        turn.text, reply?.scenarioId.orEmpty(), reply?.text.orEmpty(),
                        turn.sttMillis?.toString().orEmpty(), turn.conversationId.toString(),
                        dateFormat.format(Date(turn.createdAt)),
                    ).joinToString(",") { csv(it) },
                ).append('\n')
            }
            zip.putNextEntry(ZipEntry("manifest.csv"))
            zip.write(manifest.toString().toByteArray(Charsets.UTF_8))
            zip.closeEntry()
        }
        return turns.size
    }

    private fun csv(value: String): String =
        if (value.any { it == ',' || it == '"' || it == '\n' || it == '\r' }) "\"" + value.replace("\"", "\"\"") + "\"" else value
}
