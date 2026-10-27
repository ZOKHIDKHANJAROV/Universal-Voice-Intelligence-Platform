package uz.univoice.offline.audio

import android.content.Context
import android.media.AudioAttributes
import android.media.MediaPlayer
import java.io.File

/** Plays one clip at a time: a recorded caller turn or a pre-recorded answer. */
class Player(private val context: Context) {
    private var player: MediaPlayer? = null

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
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build(),
            )
            source()
            setOnCompletionListener { stop(); onDone() }
            setOnErrorListener { _, _, _ -> stop(); onDone(); true }
            prepare()
            start()
        }
        player = media
    }
}
