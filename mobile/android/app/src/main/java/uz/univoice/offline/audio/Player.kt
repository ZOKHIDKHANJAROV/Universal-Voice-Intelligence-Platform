package uz.univoice.offline.audio

import android.content.Context
import android.media.AudioAttributes
import android.media.MediaPlayer
import java.io.File

/** Plays one clip at a time: a recorded caller turn or a pre-recorded answer. */
class Player(
    private val context: Context,
    /** USAGE_VOICE_COMMUNICATION when answering a call: it follows the call volume and route. */
    var usage: Int = AudioAttributes.USAGE_MEDIA,
) {
    private var player: MediaPlayer? = null

    /** An external (USB) output to play through instead of the phone's speaker. */
    var device: android.media.AudioDeviceInfo? = null

    fun playFile(file: File, onDone: () -> Unit = {}) = play(onDone) { setDataSource(file.absolutePath) }

    /** Assets are stored uncompressed (noCompress "wav") so they can be opened directly. */
    fun playAsset(path: String, onDone: () -> Unit = {}) = play(onDone) {
        context.assets.openFd(path).use { setDataSource(it.fileDescriptor, it.startOffset, it.length) }
    }

    fun stop() {
        player?.run { runCatching { stop() }; release() }
        player = null
    }

    private fun play(onDone: () -> Unit, source: MediaPlayer.() -> Unit) {
        stop()
        val media = MediaPlayer().apply {
            setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(usage)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build(),
            )
            source()
            if (device != null && android.os.Build.VERSION.SDK_INT >= 28) setPreferredDevice(device)
            setOnCompletionListener { stop(); onDone() }
            setOnErrorListener { _, _, _ -> stop(); onDone(); true }
            prepare()
            start()
        }
        player = media
    }
}
