package uz.univoice.offline.scenario

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.min
import kotlin.math.roundToLong

data class Scenario(
    val id: String,
    val title: String,
    val language: String,
    val keywords: List<String>,
    val contextKeywords: List<String>,
    val action: String?,
    val message: String?,
    val enabled: Boolean,
)

data class Match(val scenario: Scenario, val confidence: Double)

/** What the bot says back: the text, its language, and the audio to play. */
data class Reply(
    val text: String,
    val language: String,
    /** Key in the prompt manifest: a scenario id, or a prompts.json key. */
    val audioKey: String,
    val scenarioId: String?,
    val confidence: Double,
    /** The caller asked for a person (an operator scenario) or was not understood. */
    val offerOperator: Boolean,
)

/**
 * Port of ScenarioService.resolve, IntentService's confidence threshold and
 * build_response (app/services, app/intent, app/voice/responses.py).
 */
class ScenarioMatcher(
    private val scenarios: List<Scenario>,
    private val prompts: Map<String, Map<String, String>>,
    private val minConfidence: Double = 0.6,
) {
    private val enabled = scenarios.filter { it.enabled }

    /**
     * Pick the scenario whose problem keywords match best. Context keywords only
     * add confidence; scenarios in [language] are tried before the others.
     */
    fun resolve(text: String, language: String? = null): Match? {
        val tokens = TextNormalizer.words(TextNormalizer.normalize(text))
        val latinTokens = TextNormalizer.words(TextNormalizer.toUzbekLatin(text))
        val groups = if (language != null) {
            listOf(enabled.filter { it.language == language }, enabled.filter { it.language != language })
        } else {
            listOf(enabled)
        }
        for (group in groups) {
            var best: Scenario? = null
            var bestScore = 0.0
            var bestProblems = 0
            var bestContext = 0
            for (scenario in group) {
                val words = if (scenario.language == "uz") latinTokens else tokens
                val problems = scenario.keywords.count { KeywordMatcher.keywordMatches(it, words) }
                if (problems == 0) continue
                val context = scenario.contextKeywords.count { KeywordMatcher.keywordMatches(it, words) }
                val score = problems + 0.25 * context
                if (score > bestScore) {
                    best = scenario; bestScore = score; bestProblems = problems; bestContext = context
                }
            }
            if (best != null) {
                val confidence = min(1.0, 0.5 + 0.15 * bestProblems + 0.05 * bestContext)
                return Match(best, (confidence * 100).roundToLong() / 100.0)
            }
        }
        return null
    }

    fun reply(text: String, callerLanguage: String): Reply {
        val match = resolve(text, callerLanguage)?.takeIf { it.confidence >= minConfidence }
        val scenario = match?.scenario
        if (scenario == null) {
            return promptReply("not_understood", callerLanguage, match?.confidence ?: 0.0)
        }
        val message = scenario.message
            ?: return promptReply("operator", scenario.language, match.confidence, scenario.id)
        return Reply(
            text = message,
            language = scenario.language,
            audioKey = scenario.id,
            scenarioId = scenario.id,
            confidence = match.confidence,
            offerOperator = scenario.action == "transfer",
        )
    }

    fun prompt(key: String, language: String): String =
        prompts[key]?.get(language) ?: prompts[key]?.values?.firstOrNull() ?: ""

    private fun promptReply(key: String, language: String, confidence: Double, scenarioId: String? = null) = Reply(
        text = prompt(key, language),
        language = language,
        audioKey = key,
        scenarioId = scenarioId,
        confidence = confidence,
        offerOperator = true,
    )

    companion object {
        fun parseScenarios(json: String): List<Scenario> {
            val array = JSONArray(json)
            return (0 until array.length()).map { i ->
                val item = array.getJSONObject(i)
                val steps = item.optJSONArray("steps")
                Scenario(
                    id = item.getString("id"),
                    title = item.optString("title"),
                    language = item.getString("language"),
                    keywords = item.optJSONArray("keywords").strings(),
                    contextKeywords = item.optJSONArray("context_keywords").strings(),
                    action = item.optString("action").takeIf { !item.isNull("action") && it.isNotEmpty() },
                    message = steps?.takeIf { it.length() > 0 }?.getJSONObject(0)?.optString("message"),
                    enabled = item.optBoolean("enabled", true),
                )
            }
        }

        fun parsePrompts(json: String): Map<String, Map<String, String>> {
            val root = JSONObject(json)
            return root.keys().asSequence().associateWith { key ->
                val byLanguage = root.getJSONObject(key)
                byLanguage.keys().asSequence().associateWith { byLanguage.getString(it) }
            }
        }

        private fun JSONArray?.strings(): List<String> =
            if (this == null) emptyList() else (0 until length()).map { getString(it) }
    }
}
