package uz.univoice.offline.call

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioManager
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import uz.univoice.offline.BuildConfig
import uz.univoice.offline.audio.Player
import uz.univoice.offline.audio.Recorder
import uz.univoice.offline.audio.Wav
import uz.univoice.offline.data.ChatStore
import uz.univoice.offline.data.PromptAudio
import uz.univoice.offline.data.Role
import uz.univoice.offline.scenario.ScenarioMatcher
import uz.univoice.offline.stt.SpeechStream
import uz.univoice.offline.stt.VoskEngine
import java.io.File
import kotlin.coroutines.resume
import kotlin.math.abs

/**
 * One answered call: greets, listens, answers with the pre-recorded audio,
 * and ends when the dialog says so. Android gives an app no way to put sound
 * into a call or to hear the caller's line, so the phone is on speaker: the
 * answers come out of its speaker into its microphone, and the caller's voice
 * comes out of the speaker into the microphone. Quality depends on the phone.
 */
class CallSession(
    private val context: Context,
    private val store: ChatStore,
    private val matcher: ScenarioMatcher,
    private val prompts: PromptAudio,
    /** Language to the model that listens in it. */
    private val engines: Map<String, VoskEngine>,
    private val dialogConfig: CallDialogConfig,
    private val caller: String,
) {
    private val player = Player(context, AudioAttributes.USAGE_VOICE_COMMUNICATION)
    private val started = SystemClock.elapsedRealtime()
    private var conversationId = 0L
    private var micWarned = false
    /** Index in [MIC_SOURCES] of the audio source that works on this phone during a call. */
    private var micSource = 0

    private fun elapsed() = SystemClock.elapsedRealtime() - started

    /** Runs until the dialog ends the call or the coroutine is cancelled (the caller hung up). */
    suspend fun run(): Outcome {
        conversationId = store.newCall(caller)
        CallState.callStarted(conversationId, caller)
        log(Role.SYSTEM, if (caller.isEmpty()) "Qoʻngʻiroq · Звонок" else "Qoʻngʻiroq · Звонок: $caller")

        val dialog = CallDialog(matcher, dialogConfig)
        var step = dialog.start()
        var callback = false
        var reason = "caller_hangup"
        try {
            while (true) {
                say(step)
                callback = callback || step.callback
                if (step.endReason != null) {
                    Log.i(TAG, "bot ends the call: ${step.endReason}")
                    reason = step.endReason
                    return Outcome(step.endReason, callback)
                }
                val heard = listen(dialog.listenMs(step, elapsed()))
                step = if (heard == null) {
                    dialog.silence(elapsed())
                } else {
                    val next = dialog.heard(heard.readings, elapsed())
                    dialog.lastHeard?.let { chosen ->
                        log(Role.CALLER, chosen.text, language = chosen.language, audioPath = heard.audio.absolutePath)
                    }
                    next
                }
            }
        } finally {
            player.stop()
            if (callback) store.markCallback(conversationId)
            log(Role.SYSTEM, "Tugadi · Завершён ($reason)")
            CallState.callEnded()
        }
    }

    data class Outcome(val reason: String?, val callback: Boolean)

    private class Turn(val readings: List<Heard>, val audio: File)

    // ----- the bot speaks -----

    private suspend fun say(step: Step) {
        for (u in step.utterances) {
            log(
                Role.BOT, u.text, language = u.language, audioKey = u.key, scenarioId = u.scenarioId,
                confidence = u.confidence,
            )
            val asset = prompts.assetFor(u.key, u.language, u.text)
            if (asset == null) {
                Log.w(TAG, "no recording for ${u.key}/${u.language}")
                continue
            }
            play(asset)
        }
    }

    private suspend fun play(asset: String) {
        withTimeoutOrNull(60_000) {
            suspendCancellableCoroutine { continuation ->
                continuation.invokeOnCancellation { player.stop() }
                runCatching { player.playAsset(asset) { if (continuation.isActive) continuation.resume(Unit) } }
                    .onFailure { e ->
                        Log.e(TAG, "playback failed", e)
                        if (continuation.isActive) continuation.resume(Unit)
                    }
            }
        }
    }

    // ----- the caller speaks -----

    /**
     * Records until the caller finishes a phrase and returns what each model
     * heard; null when nobody spoke for [timeoutMs]. Once speech starts it
     * is not cut off by the timeout.
     */
    private suspend fun listen(timeoutMs: Long): Turn? {
        val chunks = Channel<ShortArray>(Channel.UNLIMITED)
        var recorder = openMic()
        val listenStart = SystemClock.elapsedRealtime()
        try {
            recorder.start { chunks.trySend(it) }
        } catch (e: Exception) {
            warnMic("Mikrofon ochilmadi · Микрофон не открылся: ${e.message}")
            kotlinx.coroutines.delay(timeoutMs)
            return null
        }
        var streams: List<Pair<String, SpeechStream>> = openStreams()
        val audio = java.io.ByteArrayOutputStream()
        val deadline = SystemClock.elapsedRealtime() + timeoutMs
        var speechSeen = false
        var skip = SKIP_CHUNKS // the end of our own voice still in the microphone
        var count = 0
        var peak = 0
        try {
            while (true) {
                val wait = if (speechSeen) Long.MAX_VALUE else deadline - SystemClock.elapsedRealtime()
                if (wait <= 0) return null
                // Some phones give the microphone to the call and feed apps silence or nothing:
                // try the other audio sources before giving up on this turn.
                val probe = if (count == 0 || peak == 0) 1_500L else Long.MAX_VALUE
                val chunk = withTimeoutOrNull(minOf(wait, probe)) { chunks.receive() }
                if (chunk == null) {
                    val dead = SystemClock.elapsedRealtime() - listenStart >= 1_500 && (count == 0 || peak == 0)
                    if (dead && micSource < MIC_SOURCES.size - 1 && !speechSeen) {
                        recorder.stop()
                        micSource++
                        Log.w(TAG, "microphone silent; trying source ${MIC_SOURCES[micSource]}")
                        recorder = openMic()
                        recorder.start { chunks.trySend(it) }
                        count = 0; peak = 0
                        continue
                    }
                    if (deadline - SystemClock.elapsedRealtime() <= 0) return null
                    continue
                }
                if (skip-- > 0) continue
                count++
                for (sample in chunk) peak = maxOf(peak, abs(sample.toInt()))
                audio.write(chunk.toBytes())
                val ended = coroutineScope {
                    streams.map { (_, stream) -> async(Dispatchers.Default) { stream.accept(chunk) } }.awaitAll()
                }.any { it }
                if (ended) {
                    val readings = streams.map { (language, stream) -> Heard(language, stream.finish()) }
                    if (readings.any { it.text.isNotBlank() }) {
                        val file = File(store.audioDir, "call$conversationId-${System.currentTimeMillis()}.wav")
                        withContext(Dispatchers.IO) { Wav.write(file, audio.toByteArray().toShorts()) }
                        return Turn(readings, file)
                    }
                    // The recognizer's own timeout on silence: start over.
                    streams.forEach { it.second.close() }
                    streams = openStreams()
                } else if (!speechSeen && count % 3 == 0) {
                    speechSeen = streams.any { it.second.partial().isNotBlank() }
                }
            }
        } finally {
            recorder.stop()
            chunks.close()
            streams.forEach { it.second.close() }
            // Exactly zero for seconds is a muted microphone, not a quiet room.
            if ((peak == 0 && count > 30) || (count == 0 && timeoutMs > 3_000)) {
                warnMic("Mikrofon jim: telefon qoʻngʻiroq paytida mikrofonni bermayapti · Микрофон молчит: телефон не отдаёт его во время звонка")
            }
            Log.i(TAG, "listened: peak=$peak chunks=$count speech=$speechSeen")
        }
    }

    /** The microphone, or in a debug build a clip a test pushed to the phone (see [ClipMic]). */
    private fun openMic(): Mic {
        val clip = if (BuildConfig.DEBUG) context.getExternalFilesDir(null)?.resolve(TEST_CLIP)?.takeIf { it.exists() } else null
        if (clip != null) return ClipMic(clip)
        val recorder = Recorder(MIC_SOURCES[micSource])
        return object : Mic {
            override fun start(onChunk: (ShortArray) -> Unit) = recorder.start(onChunk)
            override fun stop() { recorder.stop() }
        }
    }

    private fun openStreams(): List<Pair<String, SpeechStream>> =
        engines.map { (language, engine) -> language to engine.newStream(PHRASE_END_SECONDS) }

    private fun warnMic(text: String) {
        if (micWarned) return
        micWarned = true
        log(Role.SYSTEM, text)
    }

    private fun log(
        role: Role,
        text: String,
        language: String? = null,
        audioPath: String? = null,
        audioKey: String? = null,
        scenarioId: String? = null,
        confidence: Double? = null,
    ) {
        store.add(
            conversationId, role, text, language = language, audioPath = audioPath, audioKey = audioKey,
            scenarioId = scenarioId, confidence = confidence,
        )
        CallState.callChanged()
    }

    private fun ShortArray.toBytes(): ByteArray {
        val bytes = ByteArray(size * 2)
        for (i in indices) {
            bytes[2 * i] = (this[i].toInt() and 0xff).toByte()
            bytes[2 * i + 1] = (this[i].toInt() shr 8).toByte()
        }
        return bytes
    }

    private fun ByteArray.toShorts() = ShortArray(size / 2) { i ->
        ((this[2 * i].toInt() and 0xff) or (this[2 * i + 1].toInt() shl 8)).toShort()
    }

    companion object {
        private const val TAG = "CallSession"

        /** Tried in turn when the microphone gives an app nothing during a call. */
        private val MIC_SOURCES = intArrayOf(
            android.media.MediaRecorder.AudioSource.VOICE_RECOGNITION,
            android.media.MediaRecorder.AudioSource.MIC,
            android.media.MediaRecorder.AudioSource.VOICE_COMMUNICATION,
            android.media.MediaRecorder.AudioSource.CAMCORDER,
        )

        /** Debug builds only: files/call_input.wav is played to the bot instead of the microphone, once. */
        private const val TEST_CLIP = "call_input.wav"

        /** A pause this long ends the caller's phrase. */
        private const val PHRASE_END_SECONDS = 0.8f

        /** 100 ms each: dropped after the bot stops talking, so it does not hear itself. */
        private const val SKIP_CHUNKS = 2
    }
}

/** Speaker on and the call volume up while the bot talks, put back afterwards. */
class CallAudio(context: Context) {
    private val audio = context.getSystemService(Context.AUDIO_SERVICE) as AudioManager
    private var speaker = false
    private var volume = 0

    @Suppress("DEPRECATION")
    fun enter() {
        speaker = audio.isSpeakerphoneOn
        volume = audio.getStreamVolume(AudioManager.STREAM_VOICE_CALL)
        audio.isSpeakerphoneOn = true
        audio.setStreamVolume(AudioManager.STREAM_VOICE_CALL, audio.getStreamMaxVolume(AudioManager.STREAM_VOICE_CALL), 0)
    }

    @Suppress("DEPRECATION")
    fun exit() {
        audio.isSpeakerphoneOn = speaker
        audio.setStreamVolume(AudioManager.STREAM_VOICE_CALL, volume, 0)
    }
}

private interface Mic {
    fun start(onChunk: (ShortArray) -> Unit)
    fun stop()
}

/**
 * Test aid for debug builds: plays a WAV into the call as if the caller said
 * it (then silence), so the whole dialog can be driven on an emulator whose
 * microphone is mute. The file is consumed, so only the first turn hears it.
 */
private class ClipMic(private val file: File) : Mic {
    @Volatile private var running = false

    override fun start(onChunk: (ShortArray) -> Unit) {
        val audio = Wav.readAsStt(file.inputStream())
        file.delete()
        // Half a second of quiet first: a person does not start the instant the bot stops.
        val samples = ShortArray(CHUNK * 5) + ShortArray(audio.size) { (audio[it] * 32767f).toInt().toShort() }
        running = true
        kotlin.concurrent.thread(name = "clip-mic") {
            var at = 0
            while (running) {
                val end = minOf(at + CHUNK, samples.size)
                onChunk(if (at < samples.size) samples.copyOfRange(at, end) else ShortArray(CHUNK))
                at = end.coerceAtLeast(at)
                if (at >= samples.size) at = samples.size
                Thread.sleep(100)
            }
        }
    }

    override fun stop() {
        running = false
    }

    private companion object {
        const val CHUNK = 1600
    }
}
