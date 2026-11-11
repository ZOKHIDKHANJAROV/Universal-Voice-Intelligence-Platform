package uz.univoice.offline.call

import android.accessibilityservice.AccessibilityService
import android.content.ComponentName
import android.content.Context
import android.provider.Settings
import android.view.accessibility.AccessibilityEvent

/**
 * Does nothing by itself. Since Android 10 a phone feeds silence to every
 * app that records during a call, except an app with an enabled
 * accessibility service: with this switched on, the answering machine hears
 * the caller through the speaker. It reads no screen content.
 */
class CallAccessibilityService : AccessibilityService() {
    override fun onAccessibilityEvent(event: AccessibilityEvent?) = Unit

    override fun onInterrupt() = Unit

    companion object {
        fun isEnabled(context: Context): Boolean {
            val me = ComponentName(context, CallAccessibilityService::class.java).flattenToString()
            val enabled = Settings.Secure.getString(context.contentResolver, Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES) ?: return false
            return enabled.split(':').any { it.equals(me, ignoreCase = true) }
        }
    }
}
