plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
}

// whisper.cpp is a git submodule. Without it the app still builds and runs
// (chat, scenarios, recorded answers), but speech recognition is unavailable.
val whisperDir = rootProject.file("third_party/whisper.cpp")
val withWhisper = whisperDir.resolve("CMakeLists.txt").exists()
if (!withWhisper) {
    logger.warn("third_party/whisper.cpp is missing: building without speech recognition (see mobile/android/README.md)")
}

android {
    namespace = "uz.univoice.offline"
    compileSdk = 37
    ndkVersion = "29.0.14206865"

    defaultConfig {
        applicationId = "uz.univoice.offline"
        minSdk = 26
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0"
        buildConfigField("boolean", "WITH_WHISPER", withWhisper.toString())

        ndk {
            // The Uzbek model needs a 64-bit phone with ~1.5 GB of free RAM.
            abiFilters += "arm64-v8a"
        }
        if (withWhisper) {
            externalNativeBuild {
                cmake {
                    arguments += listOf("-DWHISPER_DIR=${whisperDir.absolutePath.replace('\\', '/')}")
                    // ./gradlew assembleDebug -ParmArch=armv8.2-a+dotprod: faster on
                    // Cortex-A55/A75 and newer, crashes on older armv8.0 phones.
                    (findProperty("armArch") as String?)?.let { arguments += "-DUNIVOICE_ARM_ARCH=$it" }
                }
            }
        }
    }

    if (withWhisper) {
        externalNativeBuild {
            cmake {
                path = file("src/main/cpp/CMakeLists.txt")
                version = "4.1.2"
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
        debug {
            // whisper.cpp is unusably slow without optimizations.
            externalNativeBuild {
                cmake {
                    arguments += "-DCMAKE_BUILD_TYPE=Release"
                }
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    packaging {
        resources.excludes += "/META-INF/{AL2.0,LGPL2.1}"
    }

    androidResources {
        // MediaPlayer opens asset WAVs by file descriptor, which needs them stored uncompressed.
        noCompress += "wav"
    }
}

androidComponents {
    onVariants { variant ->
        // scenarios.json and prompts.json come straight from the server's
        // app/data, so the phone and the call bot answer the same way.
        variant.sources.assets?.addStaticSourceDirectory(rootProject.file("../../app/data").absolutePath)
    }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2026.06.00")

    implementation(composeBom)
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.10.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.10.0")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")

    testImplementation("junit:junit:4.13.2")
    // android.jar's org.json is a stub on the JVM; unit tests need the real one.
    testImplementation("org.json:json:20250517")
}
