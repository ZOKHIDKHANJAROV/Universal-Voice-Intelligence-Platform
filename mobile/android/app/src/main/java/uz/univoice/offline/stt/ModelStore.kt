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
    /** Key for its state and its custom URL; Whisper keeps the language for old settings. */
    val id: String,
    val engine: Engine,
    val language: String,
    val title: String,
    /** The downloaded file: a ggml .bin for Whisper, a .zip of the model folder for Vosk. */
    val fileName: String,
    val defaultUrl: String,
    val sizeBytes: Long?,
    /** sha256 of the file; null when it is not published yet (no check). */
    val sha256: String?,
)

object Models {
    // Official small models (Apache 2.0) from https://alphacephei.com/vosk/models.
    val VOSK_UZ = ModelSpec(
        id = "vosk_uz",
        engine = Engine.VOSK,
        language = "uz",
        title = "Oʻzbekcha (Vosk small 0.22)",
        fileName = "vosk-model-small-uz-0.22.zip",
        defaultUrl = "https://alphacephei.com/vosk/models/vosk-model-small-uz-0.22.zip",
        sizeBytes = 51_061_189,
        sha256 = "ffb77328df64f222a3fac6e368e0baff9dc1258f11b6b5ca5deaa82e5f688e6a",
    )

    val VOSK_RU = ModelSpec(
        id = "vosk_ru",
        engine = Engine.VOSK,
        language = "ru",
        title = "Русский (Vosk small 0.22)",
        fileName = "vosk-model-small-ru-0.22.zip",
        defaultUrl = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",
        sizeBytes = 46_236_750,
        sha256 = "961d5ff98a17f4aa6de69864d0aa71fa5bac682301d2b5d17a3f24c5c99a46d4",
    )

    // ZOKHID/univoice-stt-uz converted with scripts/convert_whisper_ggml.py and
    // quantized to q5_0 (whisper.cpp v1.9.5).
    val UZ = ModelSpec(
        id = "uz",
        engine = Engine.WHISPER,
        language = "uz",
        title = "Oʻzbekcha (univoice-stt-uz, medium q5_0)",
        fileName = "ggml-univoice-stt-uz-q5_0.bin",
        defaultUrl = "https://huggingface.co/ZOKHID/univoice-stt-uz-ggml/resolve/main/ggml-univoice-stt-uz-q5_0.bin",
        sizeBytes = 539_212_484,
        sha256 = "3c5b9639dac43845c6d71f49115e16caeb0009064b68418a55ecddc094c4445d",
    )

    // The Uzbek fine-tune has forgotten Russian, so Russian gets its own model.
    val RU = ModelSpec(
        id = "ru",
        engine = Engine.WHISPER,
        language = "ru",
        title = "Русский (whisper small q5_1)",
        fileName = "ggml-small-q5_1.bin",
        defaultUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin",
        sizeBytes = 190_085_487,
        sha256 = "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb",
    )

    val all = listOf(VOSK_UZ, VOSK_RU, UZ, RU)

    fun of(engine: Engine): List<ModelSpec> = all.filter { it.engine == engine }

    fun forLanguage(engine: Engine, language: String): ModelSpec =
        all.first { it.engine == engine && it.language == language }
}

/** Models live in app storage: downloaded once, or imported from a file. */
class ModelStore(private val context: Context) {
    private val directory = File(context.filesDir, "models").apply { mkdirs() }

    /** The ggml file, or the unpacked Vosk model folder. */
    fun file(spec: ModelSpec): File = File(directory, spec.fileName.removeSuffix(".zip"))

    fun isReady(spec: ModelSpec): Boolean = file(spec).let {
        when (spec.engine) {
            Engine.WHISPER -> it.isFile && it.length() > 1_000_000
            Engine.VOSK -> File(it, "am/final.mdl").isFile && File(it, "graph").isDirectory
        }
    }

    fun delete(spec: ModelSpec) {
        file(spec).deleteRecursively()
        partial(spec).delete()
    }

    /**
     * Downloads [url] into place, resuming a previous partial download.
     * [progress] gets (bytes so far, total or -1). Blocking: call off the main thread.
     */
    fun download(spec: ModelSpec, url: String, progress: (Long, Long) -> Unit) {
        // Slow mirrors stall mid-file (alphacephei.com often does): resume a few times.
        var attempt = 0
        while (true) {
            try {
                fetch(partial(spec), url, progress)
                break
            } catch (e: java.io.IOException) {
                if (++attempt >= 6) throw e
                Thread.sleep(2_000L * attempt)
            }
        }
        install(spec, partial(spec))
    }

    private fun fetch(part: File, url: String, progress: (Long, Long) -> Unit) {
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
            if (code == 416 && part.length() > 0) {
                // The previous attempt already got the whole file.
                connection.disconnect()
                break
            }
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
    }

    /** Copies a model file (or Vosk .zip) the user picked, e.g. sent over USB or a messenger. */
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
        when (spec.engine) {
            Engine.WHISPER -> check(hasMagic(part, "lmgg")) { part.delete(); "not a whisper.cpp (ggml) model file" }
            Engine.VOSK -> check(hasMagic(part, "PK\u0003\u0004")) { part.delete(); "not a Vosk model .zip" }
        }
        spec.sha256?.let { expected ->
            val actual = sha256(part)
            check(actual == expected) { part.delete(); "checksum mismatch: $actual" }
        }
        val target = file(spec)
        target.deleteRecursively()
        if (spec.engine == Engine.VOSK) {
            unzip(part, target)
            part.delete()
            check(isReady(spec)) { target.deleteRecursively(); "the archive has no Vosk model in it" }
        } else {
            check(part.renameTo(target)) { "could not move model into place" }
        }
    }

    /** Unpacks [archive] into [target], dropping the archive's top folder (vosk-model-small-uz-0.22/). */
    private fun unzip(archive: File, target: File) {
        val temporary = File(target.path + ".unpacking").apply { deleteRecursively(); mkdirs() }
        java.util.zip.ZipInputStream(archive.inputStream().buffered()).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                val path = entry.name.replace('\\', '/').substringAfter('/', "")
                if (path.isEmpty()) continue
                val out = File(temporary, path)
                check(out.canonicalPath.startsWith(temporary.canonicalPath + File.separator)) { "bad path in archive: ${entry.name}" }
                if (entry.isDirectory) out.mkdirs()
                else {
                    out.parentFile?.mkdirs()
                    out.outputStream().use { zip.copyTo(it) }
                }
            }
        }
        check(temporary.renameTo(target)) { "could not move model into place" }
    }

    // ggml: "lmgg", the magic 0x67676d6c written little-endian. zip: "PK" 3 4.
    private fun hasMagic(file: File, magic: String): Boolean = file.inputStream().use { input ->
        val head = ByteArray(4)
        input.read(head) == 4 && String(head, Charsets.ISO_8859_1) == magic
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
