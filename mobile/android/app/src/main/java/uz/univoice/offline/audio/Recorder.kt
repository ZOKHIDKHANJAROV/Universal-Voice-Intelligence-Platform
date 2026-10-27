package uz.univoice.offline.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.thread

/** Push-to-talk recording: 16 kHz mono PCM16, the rate Whisper works at. */
class Recorder {
    private var record: AudioRecord? = null
    private var worker: Thread? = null
    @Volatile private var running = false
    private val buffer = ByteArrayOutputStream()

    val isRecording: Boolean get() = running

    /** Needs the RECORD_AUDIO permission, which the UI asks for first. */
    @SuppressLint("MissingPermission")
    fun start() {
        if (running) return
        val minSize = AudioRecord.getMinBufferSize(
            Wav.STT_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
        )
        val audio = AudioRecord(
            // VOICE_RECOGNITION skips the call-oriented processing some phones apply.
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            Wav.STT_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minSize, Wav.STT_RATE),
        )
        check(audio.state == AudioRecord.STATE_INITIALIZED) { "microphone unavailable" }
        buffer.reset()
        record = audio
        running = true
        audio.startRecording()
        worker = thread(name = "recorder") {
            val chunk = ByteArray(3200) // 100 ms
            while (running) {
                val read = audio.read(chunk, 0, chunk.size)
                if (read > 0) synchronized(buffer) { buffer.write(chunk, 0, read) }
            }
        }
    }

    /** Stops and returns what was recorded. */
    fun stop(): ShortArray {
        if (!running) return ShortArray(0)
        running = false
        worker?.join(500)
        record?.run { stop(); release() }
        record = null
        val bytes = synchronized(buffer) { buffer.toByteArray() }
        val samples = ShortArray(bytes.size / 2)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(samples)
        return samples
    }
}
