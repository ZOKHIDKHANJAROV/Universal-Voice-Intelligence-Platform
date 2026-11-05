package uz.univoice.offline

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import uz.univoice.offline.call.CallDialog
import uz.univoice.offline.call.CallDialogConfig
import uz.univoice.offline.call.Heard
import uz.univoice.offline.call.Step
import uz.univoice.offline.scenario.ScenarioMatcher
import java.io.File

class CallDialogTest {
    private val repo: File = generateSequence(File("").absoluteFile) { it.parentFile }
        .first { File(it, "app/data/scenarios.json").exists() }

    private val matcher = ScenarioMatcher(
        ScenarioMatcher.parseScenarios(File(repo, "app/data/scenarios.json").readText()),
        ScenarioMatcher.parsePrompts(File(repo, "app/data/prompts.json").readText()),
    )
    private val config = CallDialogConfig(noInputMs = 15_000, followupMs = 8_000, maxCallMs = 300_000)

    private fun dialog() = CallDialog(matcher, config)
    private fun Step.keys() = utterances.map { it.key to it.language }

    @Test
    fun greetsInBothLanguagesAndListens() {
        val step = dialog().start()
        assertEquals(listOf("greeting" to "uz", "greeting" to "ru"), step.keys())
        assertNull(step.endReason)
        assertEquals(15_000, step.listenMs)
    }

    @Test
    fun answersAComplaintInItsLanguageAndOffersMoreHelp() {
        val d = dialog()
        val step = d.heard(listOf(Heard("uz", "suv chiqmayapti"), Heard("ru", "сув чик май опти")), 5_000)
        assertEquals(listOf("vending_no_water_uz" to "uz", "anything_else" to "uz"), step.keys())
        assertEquals("uz", d.callerLanguage)
        assertEquals("suv chiqmayapti", d.lastHeard?.text)
        assertEquals(8_000, step.listenMs)
        assertNull(step.endReason)
        assertFalse(step.callback)
    }

    @Test
    fun theRussianModelWinsWhenTheCallerSpeaksRussian() {
        val d = dialog()
        val step = d.heard(listOf(Heard("uz", "voda ne idet"), Heard("ru", "вода не идет")), 4_000)
        assertEquals(listOf("vending_no_water_ru" to "ru", "anything_else" to "ru"), step.keys())
        assertEquals("ru", d.lastHeard?.language)
    }

    @Test
    fun silenceAfterAnAnswerEndsTheCall() {
        val d = dialog()
        d.heard(listOf(Heard("uz", "suv chiqmayapti")), 5_000)
        val step = d.silence(30_000)
        assertEquals(listOf("goodbye" to "uz"), step.keys())
        assertEquals("done", step.endReason)
    }

    @Test
    fun silenceBeforeAnyAnswerAsksOnceThenSaysGoodbyeInBothLanguages() {
        val d = dialog()
        val ask = d.silence(15_000)
        assertEquals(listOf("no_input" to "uz", "no_input" to "ru"), ask.keys())
        assertNull(ask.endReason)
        val bye = d.silence(35_000)
        assertEquals(listOf("goodbye" to "uz", "goodbye" to "ru"), bye.keys())
        assertEquals("silence", bye.endReason)
        assertFalse(bye.callback)
    }

    @Test
    fun aRequestForAnOperatorPromisesACallback() {
        val d = dialog()
        val step = d.heard(listOf(Heard("ru", "соедините с оператором")), 6_000)
        assertEquals(listOf("callback" to "ru"), step.keys())
        assertEquals("operator", step.endReason)
        assertTrue(step.callback)
    }

    @Test
    fun theSecondMisunderstandingPromisesACallback() {
        val d = dialog()
        val first = d.heard(listOf(Heard("uz", "bugun havo yaxshi")), 5_000)
        assertEquals(listOf("repeat" to "uz", "repeat" to "ru"), first.keys())
        assertNull(first.endReason)
        val second = d.heard(listOf(Heard("uz", "men kitob oqiyman")), 12_000)
        assertEquals(listOf("callback" to "uz", "callback" to "ru"), second.keys())
        assertEquals("not_understood", second.endReason)
        assertTrue(second.callback)
    }

    @Test
    fun afterAMatchTheRepliesStayInTheCallersLanguage() {
        val d = dialog()
        d.heard(listOf(Heard("ru", "вода не идет")), 5_000)
        val step = d.heard(listOf(Heard("ru", "расскажите анекдот")), 15_000)
        assertEquals(listOf("repeat" to "ru"), step.keys())
    }

    @Test
    fun theTimeLimitEndsTheCall() {
        val d = dialog()
        val step = d.heard(listOf(Heard("uz", "suv chiqmayapti")), 300_000)
        assertEquals("time_limit", step.endReason)
        assertEquals("time_limit", d.silence(310_000).endReason)
    }

    @Test
    fun listeningNeverRunsPastTheTimeLimit() {
        val d = dialog()
        val step = d.start()
        assertEquals(15_000, d.listenMs(step, 0))
        assertEquals(4_000, d.listenMs(step, 296_000))
        assertEquals(1, d.listenMs(step, 400_000))
    }

    @Test
    fun everyPromptTheDialogUsesHasText() {
        val d = dialog()
        val steps = listOf(
            d.start(), d.silence(1), d.silence(2), d.silence(400_000),
            dialog().heard(listOf(Heard("ru", "оператор")), 1),
            dialog().heard(listOf(Heard("ru", "ничего")), 1),
        )
        for (u in steps.flatMap { it.utterances }) assertTrue("${u.key}/${u.language} has no text", u.text.isNotBlank())
    }
}
