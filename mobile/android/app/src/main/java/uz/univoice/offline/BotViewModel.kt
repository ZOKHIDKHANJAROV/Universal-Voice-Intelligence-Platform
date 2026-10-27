package uz.univoice.offline

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
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
import uz.univoice.offline.stt.Hallucinations
import uz.univoice.offline.stt.ModelSpec
import uz.univoice.offline.stt.ModelStore
import uz.univoice.offline.stt.Models
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

    // whisper.cpp contexts are not thread-safe and the Uzbek model takes ~0.6 GB
    // of RAM: one model loaded at a time, used from one thread.
    @OptIn(ExperimentalCoroutinesApi::class)
    private val sttDispatcher = Dispatchers.Default.limitedParallelism(1)
    private var engine: WhisperEngine? = null
    private var engineLanguage: String? = null
    private var recordingStartedAt = 0L

    private val _state = MutableStateFlow(UiState(language = settings.language))
    val state: StateFlow<UiState> = _state

    val sttAvailable: Boolean get() = WhisperEngine.available
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
        runCatching { recorder.start() }
            .onSuccess {
                recordingStartedAt = System.currentTimeMillis()
                _state.update { it.copy(phase = Phase.RECORDING, playingId = null) }
            }
            .onFailure { e -> _state.update { it.copy(notice = "Mikrofon: ${e.message}") } }
    }

    fun stopTalking() {
        if (_state.value.phase != Phase.RECORDING) return
        val samples = recorder.stop()
        if (samples.size < Wav.STT_RATE * 3 / 10) {
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

            val result = runCatching { transcribe(Wav.toFloat(samples), language) }
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

    private fun loadEngine(language: String): WhisperEngine {
        check(WhisperEngine.available) { "Bu ilovada nutqni tanish yoʻq · В этой сборке нет распознавания речи" }
        engine?.let { if (engineLanguage == language) return it }
        val spec = Models.forLanguage(language)
        check(models.isReady(spec)) { "Model yuklanmagan · Модель не загружена: ${spec.title}" }
        engine?.close()
        engine = null
        _state.update { it.copy(phase = Phase.LOADING_MODEL) }
        val loaded = WhisperEngine.load(models.file(spec).absolutePath)
        engine = loaded
        engineLanguage = language
        _state.update { it.copy(phase = Phase.TRANSCRIBING) }
        return loaded
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
                val current = s.models[spec.language]
                spec.language to when {
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
            if (engineLanguage == spec.language) {
                engine?.close(); engine = null; engineLanguage = null
            }
            models.delete(spec)
            setModel(spec, ModelState.Missing)
        }
    }

    private fun runModelJob(spec: ModelSpec, job: ((Long, Long) -> Unit) -> Unit) {
        if (_state.value.models[spec.language] is ModelState.Busy) return
        setModel(spec, ModelState.Busy(0, spec.sizeBytes ?: -1))
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { job { done, total -> setModel(spec, ModelState.Busy(done, total)) } }
                .onSuccess { setModel(spec, ModelState.Ready) }
                .onFailure { e -> setModel(spec, ModelState.Failed(e.message ?: e.toString())) }
        }
    }

    private fun setModel(spec: ModelSpec, value: ModelState) =
        _state.update { it.copy(models = it.models + (spec.language to value)) }

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
