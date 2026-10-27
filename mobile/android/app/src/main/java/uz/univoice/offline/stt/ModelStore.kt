package uz.univoice.offline.stt

import android.content.Context
import android.net.Uri
import java.io.File
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/** A speech model the app can use, and where to get it. */
data class ModelSpec(
    val language: String,
    val title: String,
    val fileName: String,
    val defaultUrl: String,
    val sizeBytes: Long?,
    /** sha256 of the file; null when it is not published yet (no check). */
    val sha256: String?,
)

object Models {
    // ZOKHID/univoice-stt-uz converted with scripts/convert_whisper_ggml.py and
    // quantized to q5_0 (~0.5 GB). Publish it, or import the file on the phone.
    val UZ = ModelSpec(
        language = "uz",
        title = "Oʻzbekcha (univoice-stt-uz, medium q5_0)",
        fileName = "ggml-univoice-stt-uz-q5_0.bin",
        defaultUrl = "https://huggingface.co/ZOKHID/univoice-stt-uz-ggml/resolve/main/ggml-univoice-stt-uz-q5_0.bin",
        sizeBytes = null,
        sha256 = null,
    )

    // The Uzbek fine-tune has forgotten Russian, so Russian gets its own model.
    val RU = ModelSpec(
        language = "ru",
        title = "Русский (whisper small q5_1)",
        fileName = "ggml-small-q5_1.bin",
        defaultUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin",
        sizeBytes = 190_085_487,
        sha256 = "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb",
    )

    val all = listOf(UZ, RU)

    fun forLanguage(language: String): ModelSpec = all.first { it.language == language }
}

/** Models live in app storage: downloaded once, or imported from a file. */
class ModelStore(private val context: Context) {
    private val directory = File(context.filesDir, "models").apply { mkdirs() }

    fun file(spec: ModelSpec): File = File(directory, spec.fileName)

    fun isReady(spec: ModelSpec): Boolean = file(spec).let { it.exists() && it.length() > 1_000_000 }

    fun delete(spec: ModelSpec) {
        file(spec).delete()
        partial(spec).delete()
    }

    /**
     * Downloads [url] into place, resuming a previous partial download.
     * [progress] gets (bytes so far, total or -1). Blocking: call off the main thread.
     */
    fun download(spec: ModelSpec, url: String, progress: (Long, Long) -> Unit) {
        val part = partial(spec)
        var current = URL(url)
        var redirects = 0
        while (true) {
            val connection = (current.openConnection() as HttpURLConnection).apply {
                connectTimeout = 15_000
                readTimeout = 60_000
                instanceFollowRedirects = false
                if (part.exists() && part.length() > 0) setRequestProperty("Range", "bytes=${part.length()}-")
            }
            val code = connection.responseCode
            if (code in 300..399) {
                // HuggingFace redirects to its CDN, sometimes across hosts.
                val location = connection.getHeaderField("Location") ?: error("redirect without Location")
                connection.disconnect()
                check(++redirects <= 5) { "too many redirects" }
                current = URL(current, location)
                continue
            }
            check(code == 200 || code == 206) { "HTTP $code from $current" }
            val append = code == 206
            val length = connection.contentLengthLong
            val total = if (length < 0) -1 else length + if (append) part.length() else 0
            connection.inputStream.use { input -> copy(input, part, append, total, progress) }
            connection.disconnect()
            break
        }
        install(spec, part)
    }

    /** Copies a model file the user picked (e.g. sent over USB or a messenger). */
    fun import(spec: ModelSpec, uri: Uri, progress: (Long, Long) -> Unit) {
        val part = partial(spec)
        val total = context.contentResolver.openAssetFileDescriptor(uri, "r")?.use { it.length } ?: -1
        val input = context.contentResolver.openInputStream(uri) ?: error("cannot open $uri")
        input.use { copy(it, part, append = false, total = total, progress = progress) }
        install(spec, part)
    }

    private fun partial(spec: ModelSpec) = File(directory, spec.fileName + ".part")

    private fun copy(input: InputStream, target: File, append: Boolean, total: Long, progress: (Long, Long) -> Unit) {
        var done = if (append) target.length() else 0L
        java.io.FileOutputStream(target, append).use { output ->
            val chunk = ByteArray(1 shl 16)
            var lastReport = 0L
            while (true) {
                val read = input.read(chunk)
                if (read < 0) break
                output.write(chunk, 0, read)
                done += read
                if (done - lastReport > 1 shl 20) {
                    progress(done, total)
                    lastReport = done
                }
            }
        }
        progress(done, total)
    }

    private fun install(spec: ModelSpec, part: File) {
        check(isGgml(part)) { part.delete(); "not a whisper.cpp (ggml) model file" }
        spec.sha256?.let { expected ->
            val actual = sha256(part)
            check(actual == expected) { part.delete(); "checksum mismatch: $actual" }
        }
        val target = file(spec)
        target.delete()
        check(part.renameTo(target)) { "could not move model into place" }
    }

    private fun isGgml(file: File): Boolean = file.inputStream().use { input ->
        val magic = ByteArray(4)
        // "lmgg": the ggml magic 0x67676d6c written little-endian.
        input.read(magic) == 4 && String(magic, Charsets.US_ASCII) == "lmgg"
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val chunk = ByteArray(1 shl 16)
            while (true) {
                val read = input.read(chunk)
                if (read < 0) break
                digest.update(chunk, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
