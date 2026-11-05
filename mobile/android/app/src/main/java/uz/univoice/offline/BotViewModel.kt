package uz.univoice.offline

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import uz.univoice.offline.audio.Player
import uz.univoice.offline.audio.Recorder
import uz.univoice.offline.audio.Wav
import uz.univoice.offline.data.ChatStore
import uz.univoice.offline.data.Conversation
import uz.univoice.offline.data.Exporter
import uz.univoice.offline.data.Message
import uz.univoice.offline.data.PromptAudio
import uz.univoice.offline.data.Role
import uz.univoice.offline.scenario.ScenarioMatcher
import uz.univoice.offline.stt.Engine
import uz.univoice.offline.stt.Hallucinations
import uz.univoice.offline.stt.ModelSpec
import uz.univoice.offline.stt.ModelStore
import uz.univoice.offline.stt.Models
import uz.univoice.offline.stt.SpeechEngine
import uz.univoice.offline.stt.VoskEngine
import uz.univoice.offline.stt.WhisperEngine
import java.io.File

enum class Phase { IDLE, LOADING_MODEL, RECORDING, TRANSCRIBING, PLAYING }

sealed interface ModelState {
    data object Missing : ModelState
    data class Busy(val done: Long, val total: Long) : ModelState
    data object Ready : ModelState
    data class Failed(val error: String) : ModelState
}

data class BenchmarkRow(val name: String, val audioSeconds: Double, val sttMillis: Long, val text: String)

data class UiState(
    val language: String = "uz",
    val engine: Engine = Engine.VOSK,
    val conversationId: Long = 0,
    val messages: List<Message> = emptyList(),
    val phase: Phase = Phase.IDLE,
    val models: Map<String, ModelState> = emptyMap(),
    val playingId: Long? = null,
    val notice: String? = null,
    val benchmark: List<BenchmarkRow> = emptyList(),
    val benchmarkRunning: Boolean = false,
)

class BotViewModel(application: Application) : AndroidViewModel(application) {
    private val app = application
    val settings = Settings(application)
    private val store = ChatStore(application)
    private val models = ModelStore(application)
    private val prompts = PromptAudio(application.assets)
    private val recorder = Recorder()
    private val player = Player(application)
    private val matcher: ScenarioMatcher = ScenarioMatcher(
        ScenarioMatcher.parseScenarios(asset("scenarios.json")),
        ScenarioMatcher.parsePrompts(asset("prompts.json")),
    )

    // Neither engine is thread-safe and the Whisper Uzbek model takes ~0.6 GB
    // of RAM: one model loaded at a time, used from one thread.
    @OptIn(ExperimentalCoroutinesApi::class)
    private val sttDispatcher = Dispatchers.Default.limitedParallelism(1)
    private var engine: SpeechEngine? = null
    private var engineSpec: ModelSpec? = null
    private var recordingStartedAt = 0L

    // Streaming recognition of the current recording (Vosk only).
    private var liveChunks: Channel<ShortArray>? = null
    private var liveText: Deferred<String?>? = null

    private val _state = MutableStateFlow(UiState(language = settings.language, engine = settings.engine))
    val state: StateFlow<UiState> = _state

    val sttAvailable: Boolean get() = isAvailable(_state.value.engine)

    fun isAvailable(engine: Engine): Boolean = when (engine) {
        Engine.VOSK -> VoskEngine.available
        Engine.WHISPER -> WhisperEngine.available
    }

    /** The model the chosen engine needs for [language]. */
    fun activeModel(language: String = _state.value.language): ModelSpec = Models.forLanguage(_state.value.engine, language)
    val promptCount: Int get() = prompts.count

    init {
        refreshModels()
        val last = store.conversations().firstOrNull()
        if (last != null) openConversation(last.id) else newConversation()
    }

    fun conversations(): List<Conversation> = store.conversations()

    fun setLanguage(language: String) {
        settings.language = language
        _state.update { it.copy(language = language) }
    }

    fun setEngine(engine: Engine) {
        settings.engine = engine
        _state.update { it.copy(engine = engine) }
        viewModelScope.launch(sttDispatcher) { unloadEngine() }
    }

    fun newConversation() {
        player.stop()
        val id = store.newConversation()
        val language = _state.value.language
        val greeting = matcher.prompt("greeting", language)
        val messageId = store.add(id, Role.BOT, greeting, language = language, audioKey = "greeting")
        _state.update { it.copy(conversationId = id, messages = store.messages(id)) }
        playAnswer(messageId, "greeting", language, greeting)
    }

    fun openConversation(id: Long) {
        player.stop()
        _state.update { it.copy(conversationId = id, messages = store.messages(id), playingId = null) }
    }

    fun deleteConversation(id: Long) {
        store.deleteConversation(id)
        if (id == _state.value.conversationId) {
            val next = store.conversations().firstOrNull()
            if (next != null) openConversation(next.id) else newConversation()
        }
    }

    fun dismissNotice() = _state.update { it.copy(notice = null) }

    // ----- talking -----

    fun startTalking() {
        val current = _state.value
        if (current.phase != Phase.IDLE && current.phase != Phase.PLAYING) return
        player.stop()
        val chunks = startLive(current.language)
        runCatching { recorder.start(chunks?.let { channel -> { samples: ShortArray -> channel.trySend(samples) } }) }
            .onSuccess {
                recordingStartedAt = System.currentTimeMillis()
                _state.update { it.copy(phase = Phase.RECORDING, playingId = null) }
            }
            .onFailure { e ->
                stopLive()?.cancel()
                _state.update { it.copy(notice = "Mikrofon: ${e.message}") }
            }
    }

    /**
     * Starts recognizing while the caller speaks; returns where the recorder
     * sends its audio, or null when the engine cannot stream (Whisper) or
     * the model is missing, and the recording is then recognized after release.
     */
    private fun startLive(language: String): Channel<ShortArray>? {
        val spec = activeModel(language)
        if (spec.engine != Engine.VOSK || !isAvailable(spec.engine) || !models.isReady(spec)) return null
        val chunks = Channel<ShortArray>(Channel.UNLIMITED)
        liveChunks = chunks
        liveText = viewModelScope.async(sttDispatcher) {
            runCatching {
                // Chunks wait in the channel while the model loads (first use only).
                loadEngine(language).startStream(language)?.use { stream ->
                    for (samples in chunks) stream.accept(samples)
                    stream.finish()
                }
            }.getOrNull()
        }
        return chunks
    }

    /** Ends the audio of the live recognition and returns its pending transcript. */
    private fun stopLive(): Deferred<String?>? {
        liveChunks?.close()
        liveChunks = null
        return liveText.also { liveText = null }
    }

    fun stopTalking() {
        if (_state.value.phase != Phase.RECORDING) return
        val samples = recorder.stop()
        val live = stopLive()
        if (samples.size < Wav.STT_RATE * 3 / 10) {
            live?.cancel()
            // Shorter than 0.3 s: a tap, not speech.
            _state.update { it.copy(phase = Phase.IDLE, notice = "Tugmani bosib turib gapiring · Удерживайте кнопку, пока говорите") }
            return
        }
        val conversationId = _state.value.conversationId
        val language = _state.value.language
        viewModelScope.launch {
            _state.update { it.copy(phase = Phase.TRANSCRIBING) }
            val audioFile = File(store.audioDir, "c$conversationId-${recordingStartedAt}.wav")
            withContext(Dispatchers.IO) { Wav.write(audioFile, samples) }

            val released = System.nanoTime()
            val streamed = live?.await()
            val result = if (streamed != null) {
                // Recognition time here is the wait after release: what the caller feels.
                Result.success(streamed to (System.nanoTime() - released) / 1_000_000)
            } else {
                runCatching { transcribe(Wav.toFloat(samples), language) }
            }
            val (raw, millis) = result.getOrElse { e ->
                _state.update { it.copy(phase = Phase.IDLE, notice = e.message) }
                store.add(conversationId, Role.CALLER, "", language = language, audioPath = audioFile.absolutePath)
                refreshMessages()
                return@launch
            }
            val text = if (Hallucinations.isHallucination(raw)) "" else raw
            store.add(
                conversationId, Role.CALLER, text, language = language,
                audioPath = audioFile.absolutePath, sttMillis = millis,
            )
            val reply = matcher.reply(text, language)
            val botId = store.add(
                conversationId, Role.BOT, reply.text, language = reply.language, audioKey = reply.audioKey,
                scenarioId = reply.scenarioId, confidence = reply.confidence, offerOperator = reply.offerOperator,
            )
            refreshMessages()
            playAnswer(botId, reply.audioKey, reply.language, reply.text)
        }
    }

    /** Returns the transcript and how long recognition took. */
    private suspend fun transcribe(audio: FloatArray, language: String): Pair<String, Long> =
        withContext(sttDispatcher) {
            val loaded = loadEngine(language)
            val started = System.nanoTime()
            val text = loaded.transcribe(audio, language, settings.threads)
            text to (System.nanoTime() - started) / 1_000_000
        }

    private fun loadEngine(language: String): SpeechEngine {
        val spec = activeModel(language)
        check(isAvailable(spec.engine)) { "Bu ilovada nutqni tanish yoʻq · В этой сборке нет распознавания речи" }
        engine?.let { if (engineSpec == spec) return it }
        check(models.isReady(spec)) { "Model yuklanmagan · Модель не загружена: ${spec.title}" }
        unloadEngine()
        // During live recognition the model loads while the caller is still speaking.
        _state.update { if (it.phase == Phase.RECORDING) it else it.copy(phase = Phase.LOADING_MODEL) }
        val path = models.file(spec).absolutePath
        val loaded = when (spec.engine) {
            Engine.VOSK -> VoskEngine.load(path)
            Engine.WHISPER -> WhisperEngine.load(path)
        }
        engine = loaded
        engineSpec = spec
        _state.update { if (it.phase == Phase.LOADING_MODEL) it.copy(phase = Phase.TRANSCRIBING) else it }
        return loaded
    }

    private fun unloadEngine() {
        engine?.close()
        engine = null
        engineSpec = null
    }

    // ----- playback -----

    fun play(message: Message) {
        when {
            message.role == Role.CALLER && message.audioPath != null -> {
                _state.update { it.copy(playingId = message.id) }
                player.playFile(File(message.audioPath)) { _state.update { it.copy(playingId = null) } }
            }
            message.audioKey != null -> playAnswer(message.id, message.audioKey, message.language ?: "uz", message.text)
        }
    }

    fun stopPlayback() {
        player.stop()
        _state.update { it.copy(playingId = null, phase = if (it.phase == Phase.PLAYING) Phase.IDLE else it.phase) }
    }

    fun hasAnswerAudio(message: Message): Boolean =
        message.audioKey != null && prompts.assetFor(message.audioKey, message.language ?: "uz", message.text) != null

    private fun playAnswer(messageId: Long, key: String, language: String, text: String) {
        val asset = prompts.assetFor(key, language, text)
        if (asset == null) {
            _state.update { it.copy(phase = Phase.IDLE) }
            return
        }
        _state.update { it.copy(phase = Phase.PLAYING, playingId = messageId) }
        runCatching {
            player.playAsset(asset) {
                _state.update { it.copy(phase = if (it.phase == Phase.PLAYING) Phase.IDLE else it.phase, playingId = null) }
            }
        }.onFailure { e -> _state.update { it.copy(phase = Phase.IDLE, playingId = null, notice = e.message) } }
    }

    // ----- models -----

    fun refreshModels() {
        _state.update { s ->
            s.copy(models = Models.all.associate { spec ->
                val current = s.models[spec.id]
                spec.id to when {
                    current is ModelState.Busy -> current
                    models.isReady(spec) -> ModelState.Ready
                    else -> current as? ModelState.Failed ?: ModelState.Missing
                }
            })
        }
    }

    fun download(spec: ModelSpec) = runModelJob(spec) { progress -> models.download(spec, settings.modelUrl(spec), progress) }

    fun import(spec: ModelSpec, uri: Uri) = runModelJob(spec) { progress -> models.import(spec, uri, progress) }

    fun deleteModel(spec: ModelSpec) {
        viewModelScope.launch(sttDispatcher) {
            if (engineSpec == spec) unloadEngine()
            models.delete(spec)
            setModel(spec, ModelState.Missing)
        }
    }

    private fun runModelJob(spec: ModelSpec, job: ((Long, Long) -> Unit) -> Unit) {
        if (_state.value.models[spec.id] is ModelState.Busy) return
        setModel(spec, ModelState.Busy(0, spec.sizeBytes ?: -1))
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { job { done, total -> setModel(spec, ModelState.Busy(done, total)) } }
                .onSuccess { setModel(spec, ModelState.Ready) }
                .onFailure { e -> setModel(spec, ModelState.Failed(e.message ?: e.toString())) }
        }
    }

    private fun setModel(spec: ModelSpec, value: ModelState) =
        _state.update { it.copy(models = it.models + (spec.id to value)) }

    // ----- export and benchmark -----

    /** Writes the export zip and returns it with the number of recordings in it. */
    suspend fun export(): Pair<File, Int> = withContext(Dispatchers.IO) {
        val file = File(app.cacheDir, "exports/univoice-recordings-${System.currentTimeMillis()}.zip")
        file.parentFile?.listFiles()?.forEach { it.delete() }
        file to Exporter(store).export(file)
    }

    /** Times recognition on the bundled clips and the latest recordings. */
    fun runBenchmark() {
        if (_state.value.benchmarkRunning) return
        val language = _state.value.language
        _state.update { it.copy(benchmarkRunning = true, benchmark = emptyList()) }
        viewModelScope.launch {
            val clips = buildList<Pair<String, () -> FloatArray>> {
                app.assets.list("bench")?.sorted()?.forEach { name ->
                    add("bench/$name" to { Wav.readAsStt(app.assets.open("bench/$name")) })
                }
                store.callerRecordings(limit = 10).filter { it.language == language }.forEach { m ->
                    add(File(m.audioPath!!).name to { Wav.readAsStt(File(m.audioPath).inputStream()) })
                }
            }
            for ((name, read) in clips) {
                val row = runCatching {
                    val audio = withContext(Dispatchers.IO) { read() }
                    val (text, millis) = transcribe(audio, language)
                    BenchmarkRow(name, audio.size / Wav.STT_RATE.toDouble(), millis, text)
                }.getOrElse { e -> BenchmarkRow(name, 0.0, -1, "error: ${e.message}") }
                _state.update { it.copy(benchmark = it.benchmark + row, phase = Phase.IDLE) }
            }
            _state.update { it.copy(benchmarkRunning = false, phase = Phase.IDLE) }
        }
    }

    private fun refreshMessages() =
        _state.update { it.copy(messages = store.messages(it.conversationId), phase = Phase.IDLE) }

    private fun asset(name: String) = app.assets.open(name).bufferedReader().use { it.readText() }

    override fun onCleared() {
        player.stop()
        if (recorder.isRecording) recorder.stop()
        engine?.close()
        store.close()
    }
}
