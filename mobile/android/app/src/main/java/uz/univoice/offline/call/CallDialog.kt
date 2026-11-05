package uz.univoice.offline.call

import uz.univoice.offline.scenario.ScenarioMatcher

/** One thing the bot says: a pre-recorded prompt or scenario answer. */
data class Utterance(
    /** Key in the prompt manifest: a scenario id or a prompts.json key. */
    val key: String,
    val language: String,
    val text: String,
    val scenarioId: String? = null,
    val confidence: Double? = null,
)

/** What the caller said, as one language's model heard it. */
data class Heard(val language: String, val text: String)

/**
 * What to do next: play [utterances] in order, then hang up with [endReason],
 * or (when it is null) listen for [listenMs].
 */
data class Step(
    val utterances: List<Utterance>,
    val endReason: String? = null,
    val listenMs: Long = 0,
    /** The caller's problem needs a person: the owner is to call back. */
    val callback: Boolean = false,
)

data class CallDialogConfig(
    /** Silence before the bot asks again, and then says goodbye (the server's REALTIME_NO_INPUT_SECONDS). */
    val noInputMs: Long = 15_000,
    /** Silence after an answer and "anything else?" before goodbye (REALTIME_FOLLOWUP_SECONDS). */
    val followupMs: Long = 8_000,
    /** Longest call; then the bot says goodbye and hangs up (REALTIME_MAX_CALL_SECONDS). */
    val maxCallMs: Long = 300_000,
)

/**
 * The conversation of an answered call, like AudioSocketHandler on the server
 * but without an operator to transfer to: a request for one, or two replies
 * that were not understood, end with "we recorded your request, an operator
 * will call you back". Pure logic: no audio, no clock; the caller passes the
 * time since the call began and plays the steps.
 */
class CallDialog(
    private val matcher: ScenarioMatcher,
    private val config: CallDialogConfig = CallDialogConfig(),
) {
    /** The caller's language, once a scenario matched or they have been heard twice. */
    var callerLanguage: String? = null
        private set

    /** The reading of the last turn that [heard] chose. */
    var lastHeard: Heard? = null
        private set

    private var answered = false
    private var reprompted = false
    private var misunderstood = 0

    fun start(): Step = Step(prompts("greeting"), listenMs = config.noInputMs)

    /** The caller said something; [candidates] are the readings, one per language model. */
    fun heard(candidates: List<Heard>, elapsedMs: Long): Step {
        val spoken = candidates.filter { it.text.isNotBlank() }
        if (spoken.isEmpty()) return silence(elapsedMs)
        if (elapsedMs >= config.maxCallMs) return timeLimit()

        val replies = spoken.map { it to matcher.reply(it.text, it.language) }
        // A matched scenario beats none, then more confidence, then a model
        // hearing the scenario's own language (Russian words read by the
        // Uzbek model can still hit an Uzbek keyword by accident).
        val (heard, reply) = replies.maxWithOrNull(
            compareBy({ it.second.scenarioId != null }, { it.second.confidence },
                { it.second.scenarioId != null && it.second.language == it.first.language }),
        )!!
        lastHeard = heard

        if (reply.scenarioId != null) {
            callerLanguage = reply.language
            if (reply.audioKey == "operator" || reply.offerOperator) return callback("operator", reply.language)
            answered = true
            return Step(
                listOf(
                    Utterance(reply.audioKey, reply.language, reply.text, reply.scenarioId, reply.confidence),
                    prompt("anything_else", reply.language),
                ),
                listenMs = config.followupMs,
            )
        }

        // Nothing matched, so the language is still a guess: speak both.
        return if (++misunderstood < 2) {
            Step(prompts("repeat"), listenMs = config.noInputMs)
        } else {
            callback("not_understood", callerLanguage)
        }
    }

    /** Nothing was said for as long as the last step allowed. */
    fun silence(elapsedMs: Long): Step = when {
        elapsedMs >= config.maxCallMs -> timeLimit()
        answered -> Step(prompts("goodbye"), endReason = "done")
        !reprompted -> {
            reprompted = true
            Step(prompts("no_input"), listenMs = config.noInputMs)
        }
        else -> Step(prompts("goodbye"), endReason = "silence")
    }

    /** How long to listen after [step], so the call never outlives its limit. */
    fun listenMs(step: Step, elapsedMs: Long): Long =
        step.listenMs.coerceAtMost((config.maxCallMs - elapsedMs).coerceAtLeast(1))

    private fun timeLimit() = Step(prompts("time_limit"), endReason = "time_limit")

    private fun callback(reason: String, language: String?) =
        Step(prompts("callback", language), endReason = reason, callback = true)

    /** In the caller's language, or in both while it is not known. */
    private fun prompts(key: String, language: String? = callerLanguage): List<Utterance> =
        language?.let { listOf(prompt(key, it)) } ?: listOf(prompt(key, "uz"), prompt(key, "ru"))

    private fun prompt(key: String, language: String) = Utterance(key, language, matcher.prompt(key, language))
}
