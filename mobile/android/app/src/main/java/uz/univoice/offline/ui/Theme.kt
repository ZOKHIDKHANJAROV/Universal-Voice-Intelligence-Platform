package uz.univoice.offline.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/** The logo's two colors: teal into blue. */
val BrandTeal = Color(0xFF0D9488)
val BrandBlue = Color(0xFF1D4ED8)

private val light = lightColorScheme(
    primary = Color(0xFF0F766E),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFCCFBF1),
    onPrimaryContainer = Color(0xFF042F2E),
    secondary = Color(0xFF1D4ED8),
    secondaryContainer = Color(0xFFDBEAFE),
    onSecondaryContainer = Color(0xFF172554),
    tertiaryContainer = Color(0xFFFEF3C7),
    onTertiaryContainer = Color(0xFF451A03),
    background = Color(0xFFF6FAFA),
    surface = Color(0xFFF6FAFA),
    surfaceVariant = Color(0xFFE3EEEE),
)

private val dark = darkColorScheme(
    primary = Color(0xFF5EEAD4),
    onPrimary = Color(0xFF003731),
    primaryContainer = Color(0xFF134E4A),
    onPrimaryContainer = Color(0xFFCCFBF1),
    secondary = Color(0xFF93C5FD),
    secondaryContainer = Color(0xFF1E3A8A),
    onSecondaryContainer = Color(0xFFDBEAFE),
    tertiaryContainer = Color(0xFF78350F),
    onTertiaryContainer = Color(0xFFFEF3C7),
    background = Color(0xFF0B1415),
    surface = Color(0xFF0B1415),
    surfaceVariant = Color(0xFF1B2A2B),
)

@Composable
fun UniVoiceTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = if (isSystemInDarkTheme()) dark else light, content = content)
}
