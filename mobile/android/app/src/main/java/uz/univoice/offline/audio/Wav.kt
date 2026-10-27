package uz.univoice.offline.audio

import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Mono 16-bit PCM WAV files and the float audio whisper.cpp expects. */
object Wav {
    const val STT_RATE = 16_000

    fun write(file: File, samples: ShortArray, sampleRate: Int = STT_RATE) {
        val data = ByteBuffer.allocate(samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        samples.forEach { data.putShort(it) }
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray()); putInt(36 + data.capacity()); put("WAVE".toByteArray())
            put("fmt ".toByteArray()); putInt(16); putShort(1); putShort(1)
            putInt(sampleRate); putInt(sampleRate * 2); putShort(2); putShort(16)
            put("data".toByteArray()); putInt(data.capacity())
        }
        file.parentFile?.mkdirs()
        file.outputStream().use { it.write(header.array()); it.write(data.array()) }
    }

    /** Reads a PCM16 WAV (any channels, any rate) as mono floats at 16 kHz. */
    fun readAsStt(input: InputStream): FloatArray {
        val bytes = input.use { it.readBytes() }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(bytes.size >= 12 && String(bytes, 0, 4) == "RIFF" && String(bytes, 8, 4) == "WAVE") { "not a WAV file" }
        var channels = 1
        var rate = STT_RATE
        var bits = 16
        var offset = 12
        while (offset + 8 <= bytes.size) {
            val id = String(bytes, offset, 4)
            val size = buffer.getInt(offset + 4)
            val body = offset + 8
            if (id == "fmt ") {
                channels = buffer.getShort(body + 2).toInt()
                rate = buffer.getInt(body + 4)
                bits = buffer.getShort(body + 14).toInt()
            } else if (id == "data") {
                require(bits == 16) { "only 16-bit PCM WAV is supported" }
                val frames = minOf(size, bytes.size - body) / (2 * channels)
                val mono = FloatArray(frames) { i ->
                    var sum = 0f
                    for (c in 0 until channels) sum += buffer.getShort(body + (i * channels + c) * 2)
                    sum / channels / 32768f
                }
                return resample(mono, rate, STT_RATE)
            }
            offset = body + size + (size and 1)
        }
        error("WAV file has no data chunk")
    }

    fun toFloat(samples: ShortArray): FloatArray = FloatArray(samples.size) { samples[it] / 32768f }

    /** Linear interpolation: good enough for speech going into Whisper. */
    fun resample(audio: FloatArray, from: Int, to: Int): FloatArray {
        if (from == to || audio.isEmpty()) return audio
        val out = FloatArray((audio.size.toLong() * to / from).toInt())
        val step = from.toDouble() / to
        for (i in out.indices) {
            val position = i * step
            val left = position.toInt().coerceAtMost(audio.size - 1)
            val right = (left + 1).coerceAtMost(audio.size - 1)
            val fraction = (position - left).toFloat()
            out[i] = audio[left] * (1 - fraction) + audio[right] * fraction
        }
        return out
    }
}
