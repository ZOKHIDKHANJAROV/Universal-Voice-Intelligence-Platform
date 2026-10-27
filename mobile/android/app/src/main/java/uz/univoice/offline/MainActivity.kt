package uz.univoice.offline

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import uz.univoice.offline.ui.BenchmarkScreen
import uz.univoice.offline.ui.ChatScreen
import uz.univoice.offline.ui.HistoryScreen
import uz.univoice.offline.ui.ModelsScreen
import uz.univoice.offline.ui.Screen
import uz.univoice.offline.ui.SettingsScreen

class MainActivity : ComponentActivity() {
    private val bot: BotViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MaterialTheme(colorScheme = if (isSystemInDarkTheme()) darkColorScheme() else lightColorScheme()) {
                val state by bot.state.collectAsStateWithLifecycle()
                var screen by rememberSaveable { mutableStateOf(Screen.CHAT) }
                BackHandler(enabled = screen != Screen.CHAT) { screen = Screen.CHAT }
                val back = { screen = Screen.CHAT }
                when (screen) {
                    Screen.CHAT -> ChatScreen(bot, state, onNavigate = { screen = it })
                    Screen.HISTORY -> HistoryScreen(bot, state, onBack = back)
                    Screen.MODELS -> ModelsScreen(bot, state, onBack = back)
                    Screen.BENCHMARK -> BenchmarkScreen(bot, state, onBack = back)
                    Screen.SETTINGS -> SettingsScreen(bot, onBack = back)
                }
            }
        }
    }

    override fun onStop() {
        super.onStop()
        // Releasing the button off-screen never reaches the UI; stop here instead.
        bot.stopTalking()
    }
}
