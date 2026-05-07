package com.group45.smolvlm

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.speech.RecognizerIntent
import android.speech.tts.TextToSpeech
import android.util.Log
import android.os.Vibrator
import android.os.VibrationEffect
import android.widget.VideoView
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.core.content.ContextCompat
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.platform.LocalHapticFeedback
import androidx.compose.ui.hapticfeedback.HapticFeedbackType
import androidx.compose.ui.semantics.LiveRegionMode
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.semantics.contentDescription
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.util.Locale

// ── Theme Palette ──────────────────────────────────────────────────────────────

/**
 * Defines the color palette for the application's themes.
 */
object Palette {
    val WhiteAccent = Color(0xFFFFFFFF)
    val DarkBg      = Color(0xFF0D0D0D)
    val DarkCard    = Color(0xFF1A1A1A)
    val DarkCard2   = Color(0xFF222222)
    val GrayText    = Color(0xFF888888)
    val LightBg     = Color(0xFFF4F6F8)
    val LightCard   = Color(0xFFFFFFFF)
    val LightCard2  = Color(0xFFEEF0F3)
    val LightText   = Color(0xFF1A1A2E)
    val LightSub    = Color(0xFF666680)
    val SystemBg    = Color(0xFF1E2130)
    val SystemCard  = Color(0xFF252A3B)
    val SystemCard2 = Color(0xFF2E3348)
    val SystemText  = Color(0xFFE0E4F0)
    val SystemSub   = Color(0xFF8890AA)
}

/**
 * Holds color configuration for various UI elements in a specific theme.
 */
data class ThemeColors(
    val bg: Color, val card: Color, val card2: Color,
    val text: Color, val sub: Color, val accent: Color, val onAccent: Color
)

/**
 * Resolves [ThemeColors] based on the selected [AppTheme].
 */
fun themeColors(theme: AppTheme): ThemeColors = when (theme) {
    AppTheme.DARK   -> ThemeColors(Palette.DarkBg,   Palette.DarkCard,   Palette.DarkCard2,   Color.White,        Palette.GrayText,  Palette.WhiteAccent, Color(0xFF0D0D0D))
    AppTheme.LIGHT  -> ThemeColors(Palette.LightBg,  Palette.LightCard,  Palette.LightCard2,  Palette.LightText,  Palette.LightSub,  Color(0xFF1A7A4A),   Color.White)
    AppTheme.SYSTEM -> ThemeColors(Palette.SystemBg, Palette.SystemCard, Palette.SystemCard2, Palette.SystemText, Palette.SystemSub, Color(0xFF5B8DEF),   Color.White)
}

// ── Activity ───────────────────────────────────────────────────────────────────

/**
 * Triggers a short haptic vibration feedback.
 */
fun vibrate(context: Context) {
    val vibrator = context.getSystemService(Context.VIBRATOR_SERVICE) as android.os.Vibrator
    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
        vibrator.vibrate(android.os.VibrationEffect.createOneShot(50, android.os.VibrationEffect.DEFAULT_AMPLITUDE))
    } else {
        @Suppress("DEPRECATION")
        vibrator.vibrate(50)
    }
}

/**
 * The main activity of the application, responsible for initializing Text-to-Speech
 * and setting up the Compose UI.
 */
class MainActivity : ComponentActivity(), TextToSpeech.OnInitListener {
    private lateinit var tts: TextToSpeech
    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)
        tts = TextToSpeech(this, this)
        setContent { SmolVLMApp(tts) }
    }
    override fun onInit(status: Int) { if (status == TextToSpeech.SUCCESS) tts.language = Locale.US }
    override fun onDestroy() { tts.shutdown(); super.onDestroy() }
}

// ── App Navigation ─────────────────────────────────────────────────────────────

/**
 * Represents the different screens available in the app navigation.
 */
enum class Screen(val label: String, val icon: androidx.compose.ui.graphics.vector.ImageVector) {
    HOME("Home", Icons.Default.Home),
    LIVE("Live", Icons.Default.PlayArrow),
    HISTORY("History", Icons.Default.List),
    COMMUNITY("Community", Icons.Default.Info),
    SETTINGS("Settings", Icons.Default.Settings)
}

// ── Main App ───────────────────────────────────────────────────────────────────

/**
 * Root composable for the SmolVLM application.
 *
 * @param tts The TextToSpeech instance for voice feedback.
 * @param vm The ViewModel handling inference and app state.
 */
@OptIn(ExperimentalMaterial3Api::class, ExperimentalComposeUiApi::class)
@Composable
fun SmolVLMApp(tts: TextToSpeech, vm: InferenceViewModel = viewModel()) {
    val appTheme by vm.appTheme.collectAsState()
    val tc = themeColors(appTheme)

    MainContent(tts, vm, tc)
}

/**
 * Main content area of the app, containing the Scaffold and navigation logic.
 */
@OptIn(ExperimentalComposeUiApi::class, ExperimentalComposeUiApi::class)
@Composable
fun MainContent(tts: TextToSpeech, vm: InferenceViewModel, tc: ThemeColors) {
    val context = LocalContext.current
    var currentScreen by remember {
        mutableStateOf(Screen.HOME) 
    }

    val navItems = Screen.entries.toList()

    Scaffold(
        bottomBar = {
            NavigationBar(
                containerColor = tc.card,
                tonalElevation = 8.dp
            ) {
                navItems.forEach { screen ->
                    NavigationBarItem(
                        selected = currentScreen == screen,
                        onClick = {
                            vibrate(context)
                            currentScreen = screen
                        },
                        label = { Text(screen.label, fontSize = 10.sp) },
                        icon = { Icon(screen.icon, contentDescription = screen.label) },
                        colors = NavigationBarItemDefaults.colors(
                            selectedIconColor = tc.accent,
                            selectedTextColor = tc.accent,
                            unselectedIconColor = tc.sub,
                            unselectedTextColor = tc.sub,
                            indicatorColor = tc.accent.copy(alpha = 0.1f)
                        )
                    )
                }
            }
        }
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .padding(innerPadding)
                .fillMaxSize()
                .background(tc.bg)
        ) {
            when (currentScreen) {
                Screen.HOME -> HomeScreen(tts, vm, tc)
                Screen.LIVE -> LiveScreen(tts, vm, tc)
                Screen.HISTORY -> HistoryScreen(vm, tc, onNavigateToHome = { currentScreen = Screen.HOME })
                Screen.COMMUNITY -> CommunityScreen(tc)
                Screen.SETTINGS -> SettingsScreen(vm, tc)
            }
        }
    }
}

// ── Home Screen ───────────────────────────────────────────────────────────────

/**
 * Home screen for analyzing images and videos from the gallery or camera.
 */
@OptIn(ExperimentalMaterial3Api::class, ExperimentalComposeUiApi::class)
@Composable
fun HomeScreen(tts: TextToSpeech, vm: InferenceViewModel, tc: ThemeColors) {
    val state by vm.homeState.collectAsState()
    val videoUri by vm.selectedVideo.collectAsState()
    val imageUri by vm.selectedImage.collectAsState()
    val qualityMode by vm.qualityMode.collectAsState()
    val progress by vm.homeProgress.collectAsState()
    val question by vm.homeQuestion.collectAsState()

    val context = LocalContext.current
    val focusManager = LocalFocusManager.current
    val focusRequester = remember { FocusRequester() }
    val haptic = LocalHapticFeedback.current

    var isPlaying by remember { mutableStateOf(false) }
    var showQualitySheet by remember { mutableStateOf(false) }
    var pendingCameraUri by remember { mutableStateOf<Uri?>(null) }

    // Launchers
    val speechLauncher = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { r ->
        r.data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)?.firstOrNull()?.let { vm.setHomeQuestion(it) }
    }
    val imagePicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        tts.stop()
        uri?.let { u ->
            context.contentResolver.takePersistableUriPermission(u, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            val mimeType = context.contentResolver.getType(u) ?: ""
            if (mimeType.startsWith("video/")) {
                isPlaying = true; vm.setVideo(u); vm.setHomeQuestion("")
            } else {
                vm.setImage(u)
            }
        }
    }
    val cameraCaptureLauncher = rememberLauncherForActivityResult(ActivityResultContracts.TakePicture()) { success ->
        tts.stop()
        if (success) pendingCameraUri?.let { uri -> vm.setImage(uri) }
    }
    var pendingVideoUri by remember { mutableStateOf<Uri?>(null) }
    val cameraVideoLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CaptureVideo()) { success ->
        tts.stop()
        if (success) pendingVideoUri?.let { uri -> isPlaying = true; vm.setVideo(uri); vm.setHomeQuestion("") }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .clickable(indication = null, interactionSource = remember { MutableInteractionSource() }) { focusManager.clearFocus() }
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 20.dp, vertical = 24.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp)
    ) {
        // Header
        Column {
            Text(
                text = "Hi, Welcome back!",
                fontSize = 14.sp,
                color = tc.sub,
                fontWeight = FontWeight.Medium
            )
            Text(
                text = "Mozix.ai",
                fontSize = 32.sp,
                fontWeight = FontWeight.Black,
                letterSpacing = 2.sp,
                color = tc.accent,
                modifier = Modifier.padding(bottom = 8.dp)
            )
        }

        // Quality & Mode Selector
        Surface(
            color = tc.card,
            shape = RoundedCornerShape(16.dp),
            border = BorderStroke(1.5.dp, tc.accent.copy(alpha = 0.4f)),
            modifier = Modifier.fillMaxWidth().clickable {
                vibrate(context)
                showQualitySheet = true
            }
        ) {
            Row(
                modifier = Modifier.padding(16.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(qualityMode.emoji, fontSize = 24.sp)
                    Spacer(Modifier.width(12.dp))
                    Column {
                        Text(qualityMode.label, color = tc.text, fontWeight = FontWeight.Bold, fontSize = 16.sp)
                        Text(qualityMode.description, color = tc.sub, fontSize = 12.sp)
                    }
                }
                Icon(Icons.Default.KeyboardArrowRight, null, tint = tc.accent, modifier = Modifier.size(24.dp))
            }
        }

        // Media Display
        if (videoUri != null) {
            VideoPlayerSection(videoUri!!, isPlaying, onPlayingChange = { isPlaying = it }, tc, imagePicker, context, cameraCaptureLauncher, cameraVideoLauncher, { pendingCameraUri = it }, { pendingVideoUri = it })
        } else if (imageUri != null) {
            ImagePreviewSection(imageUri!!, tc, context, imagePicker, cameraCaptureLauncher, cameraVideoLauncher, { pendingCameraUri = it }, { pendingVideoUri = it })
        } else {
            MediaPickerSection(tc, imagePicker, context, cameraCaptureLauncher, cameraVideoLauncher, { pendingCameraUri = it }, { pendingVideoUri = it })
        }

        // Question Input
        OutlinedTextField(
            value = question,
            onValueChange = { vm.setHomeQuestion(it) },
            modifier = Modifier.fillMaxWidth().focusRequester(focusRequester),
            label = { Text("Ask about this media…", fontSize = 16.sp) },
            maxLines = 4,
            textStyle = androidx.compose.ui.text.TextStyle(fontSize = 18.sp, color = tc.text),
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
            keyboardActions = KeyboardActions(onDone = { focusManager.clearFocus() }),
            trailingIcon = {
                if (question.isNotEmpty()) {
                    IconButton(onClick = {
                        vibrate(context)
                        vm.setHomeQuestion(""); focusManager.clearFocus()
                    }) {
                        Icon(Icons.Default.Clear, null, tint = tc.sub)
                    }
                } else {
                    IconButton(onClick = {
                        vibrate(context)
                        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                        }
                        speechLauncher.launch(intent)
                    }) { Icon(Icons.Filled.Mic, null, tint = tc.accent) }
                }
            },
            colors = OutlinedTextFieldDefaults.colors(
                focusedTextColor = tc.text, unfocusedTextColor = tc.text,
                cursorColor = tc.accent,
                focusedBorderColor = tc.accent, unfocusedBorderColor = tc.sub.copy(alpha = 0.4f),
                focusedLabelColor = tc.accent, unfocusedLabelColor = tc.sub
            ),
            shape = RoundedCornerShape(12.dp)
        )

        // Quick Chips
        Row(
            modifier = Modifier.horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            listOf("What is happening?", "Describe scene", "Identify objects", "Read text").forEach { q ->
                AssistChip(
                    onClick = {
                        vibrate(context)
                        vm.setHomeQuestion(q)
                    },
                    label = { Text(q, fontSize = 12.sp) },
                    colors = AssistChipDefaults.assistChipColors(containerColor = tc.card2, labelColor = tc.text)
                )
            }
        }

        // Action Button
        if (state is UiState.Loading) {
            HomeLoadingSection(state as UiState.Loading, progress, tc, vm, focusManager)
        } else {
            val isEnabled = (videoUri != null || imageUri != null) && question.isNotBlank()
            Button(
                onClick = {
                    vibrate(context)
                    focusManager.clearFocus()
                    tts.stop()
                    vm.runInference(question)
                },
                modifier = Modifier.fillMaxWidth().height(56.dp),
                enabled = isEnabled,
                shape = RoundedCornerShape(16.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = tc.accent,
                    contentColor = tc.onAccent,
                    disabledContainerColor = tc.accent.copy(alpha = 0.2f),
                    disabledContentColor = tc.onAccent.copy(alpha = 0.5f)
                )
            ) {
                Icon(Icons.Default.Search, null)
                Spacer(Modifier.width(12.dp))
                Text("Analyse", fontWeight = FontWeight.Bold, fontSize = 18.sp)
            }
        }

        // Result Section
        if (state is UiState.Done) {
            HomeResultSection(vm, (state as UiState.Done).result, question, tc, context, tts) { 
                vm.setHomeQuestion("")
                focusRequester.requestFocus()
            }
        }

        // Error Section
        if (state is UiState.Failure) {
            Surface(color = Color.Red.copy(alpha = 0.1f), shape = RoundedCornerShape(12.dp), border = BorderStroke(1.dp, Color.Red)) {
                Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.Warning, null, tint = Color.Red)
                    Spacer(Modifier.width(12.dp))
                    Text((state as UiState.Failure).message, color = tc.text, fontSize = 14.sp)
                }
            }
        }

        Spacer(Modifier.height(40.dp))
    }

    // Quality Sheet
    if (showQualitySheet) {
        QualityDialog(qualityMode, tc, onDismiss = { showQualitySheet = false }) { vm.setQualityMode(it) }
    }
}

// ── Live Screen ───────────────────────────────────────────────────────────────

/**
 * Live vision screen that uses the camera to provide real-time narration and analysis.
 */
@OptIn(ExperimentalMaterial3Api::class, ExperimentalComposeUiApi::class)
@Composable
fun LiveScreen(tts: TextToSpeech, vm: InferenceViewModel, tc: ThemeColors) {
    val state by vm.liveState.collectAsState()
    val progress by vm.liveProgress.collectAsState()
    val isNarrating by vm.isNarrating.collectAsState()
    val isScanning by vm.isNarratorScanning.collectAsState()
    val narratorText by vm.narratorText.collectAsState()

    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val focusManager = LocalFocusManager.current
    val haptic = LocalHapticFeedback.current
    val density = LocalDensity.current

    // Pulsing animation for Narrator
    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val alpha by infiniteTransition.animateFloat(
        initialValue = 0.5f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "alpha"
    )
    val scale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 1.05f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "scale"
    )

    var question by remember { mutableStateOf("") }
    val previewView = remember { PreviewView(context) }
    var imageCapture: ImageCapture? by remember { mutableStateOf(null) }
    var camera: Camera? by remember { mutableStateOf(null) }
    var isFlashOn by remember { mutableStateOf(false) }
    var zoomLevel by remember { mutableStateOf(0f) }
    var hasCameraPermission by remember { mutableStateOf(false) }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasCameraPermission = isGranted
    }

    val speechLauncher = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { r ->
        r.data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)?.firstOrNull()?.let { question = it }
    }

    LaunchedEffect(Unit) {
        if (ContextCompat.checkSelfPermission(context, android.Manifest.permission.CAMERA) == android.content.pm.PackageManager.PERMISSION_GRANTED) {
            hasCameraPermission = true
        } else {
            permissionLauncher.launch(android.Manifest.permission.CAMERA)
        }
    }

    // Start Camera
    LaunchedEffect(hasCameraPermission) {
        if (!hasCameraPermission) return@LaunchedEffect
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            try {
                cameraProvider.unbindAll()
                camera = cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, imageCapture)
            } catch (e: Exception) {
                Log.e("LiveScreen", "Use case binding failed", e)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    LaunchedEffect(isFlashOn, camera) {
        camera?.cameraControl?.enableTorch(isFlashOn)
    }

    LaunchedEffect(zoomLevel, camera) {
        camera?.cameraControl?.setLinearZoom(zoomLevel)
    }

    val scope = rememberCoroutineScope()

    Box(Modifier.fillMaxSize()) {
        if (!hasCameraPermission) {
            Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("Camera permission is required for Live Mode", color = tc.text)
            }
        } else {
            // Camera Preview
            AndroidView(
                factory = { previewView },
                modifier = Modifier
                    .fillMaxSize()
                    .semantics { contentDescription = "Live camera feed" }
            )

            // Scanning Line Animation for Narrator
            if (isScanning) {
                val scanTransition = rememberInfiniteTransition(label = "scan")
                val translateY by scanTransition.animateFloat(
                    initialValue = 0f,
                    targetValue = 1f,
                    animationSpec = infiniteRepeatable(
                        animation = tween(1500, easing = LinearEasing),
                        repeatMode = RepeatMode.Restart
                    ),
                    label = "translateY"
                )

                Box(Modifier.fillMaxSize()) {
                    Canvas(Modifier.fillMaxSize()) {
                        val y = size.height * translateY
                        drawLine(
                            color = tc.accent.copy(alpha = 0.6f),
                            start = Offset(0f, y),
                            end = Offset(size.width, y),
                            strokeWidth = with(density) { 4.dp.toPx() }
                        )
                    }
                    Box(
                        Modifier
                            .fillMaxWidth()
                            .height(100.dp)
                            .graphicsLayer {
                                translationY = (previewView.height * translateY) - with(density) { 50.dp.toPx() }
                            }
                            .background(
                                Brush.verticalGradient(
                                    listOf(Color.Transparent, tc.accent.copy(alpha = 0.2f), Color.Transparent)
                                )
                            )
                    )
                }
            }
        }

        // Zoom Slider Overlay (Centered on the right edge)
        if (hasCameraPermission && state !is UiState.Done) {
            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .padding(end = 2.dp)
                    .offset(x=9.dp)// Optimized distance from edge
                    .align(Alignment.CenterEnd),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    modifier = Modifier.wrapContentWidth(unbounded = true),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(2.dp)
                ) {
                    Icon(
                        Icons.Default.Add,
                        contentDescription = "Zoom In",
                        tint = Color.White,
                        modifier = Modifier.size(18.dp)
                    )
                    
                    Slider(
                        value = zoomLevel,
                        onValueChange = { zoomLevel = it },
                        modifier = Modifier
                            .graphicsLayer {
                                rotationZ = 270f
                                transformOrigin = androidx.compose.ui.graphics.TransformOrigin.Center
                            }
                            .width(180.dp)
                            .height(32.dp),
                        colors = SliderDefaults.colors(
                            thumbColor = tc.accent,
                            activeTrackColor = tc.accent,
                            inactiveTrackColor = Color.White.copy(alpha = 0.5f)
                        )
                    )

                    Icon(
                        Icons.Default.Remove,
                        contentDescription = "Zoom Out",
                        tint = Color.White,
                        modifier = Modifier.size(18.dp)
                    )
                }
            }
        }

        // Narrator Text Overlay
        if (isNarrating && narratorText.isNotEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 24.dp, vertical = 120.dp),
                contentAlignment = Alignment.Center
            ) {
                Surface(
                    color = Color.Black.copy(alpha = 0.7f),
                    shape = RoundedCornerShape(20.dp),
                    border = BorderStroke(1.dp, tc.accent.copy(alpha = 0.5f))
                ) {
                    Box(Modifier.padding(16.dp)) {
                        Text(
                            text = narratorText,
                            color = Color.White,
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold,
                            textAlign = TextAlign.Center,
                            lineHeight = 24.sp,
                            modifier = Modifier.padding(end = 24.dp)
                        )
                        IconButton(
                            onClick = { 
                                vibrate(context)
                                vm.clearNarratorText() 
                            },
                            modifier = Modifier
                                .align(Alignment.TopEnd)
                                .size(24.dp)
                                .offset(x = 12.dp, y = (-12).dp)
                        ) {
                            Icon(
                                imageVector = Icons.Default.Close,
                                contentDescription = "Clear Text",
                                tint = Color.White.copy(alpha = 0.7f),
                                modifier = Modifier.size(16.dp)
                            )
                        }
                    }
                }
            }
        }

        // UI Overlay
        Column(
            modifier = Modifier.fillMaxSize().padding(20.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            // Top Header
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Surface(
                        color = if (state is UiState.Loading) Color.Red.copy(alpha = 0.8f) else Color.Black.copy(alpha = 0.5f),
                        shape = RoundedCornerShape(12.dp)
                    ) {
                        Text(
                            if (state is UiState.Loading) "AI ANALYSING…" else "LIVE VISION · READY",
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                            color = Color.White,
                            fontWeight = FontWeight.Bold,
                            fontSize = 12.sp
                        )
                    }

                    // Flash Toggle
                    Surface(
                        color = if (isFlashOn) Color.Yellow.copy(alpha = 0.8f) else Color.Black.copy(alpha = 0.5f),
                        shape = CircleShape,
                        modifier = Modifier
                            .size(36.dp)
                            .clickable {
                                vibrate(context)
                                isFlashOn = !isFlashOn
                            }
                    ) {
                        Box(contentAlignment = Alignment.Center) {
                            Icon(
                                imageVector = if (isFlashOn) Icons.Default.FlashOn else Icons.Default.FlashOff,
                                contentDescription = "Flash Toggle",
                                tint = if (isFlashOn) Color.Black else Color.White,
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }
                }

                // Narrator Toggle
                Surface(
                    color = if (isNarrating) tc.accent.copy(alpha = alpha) else Color.Black.copy(alpha = 0.5f),
                    shape = RoundedCornerShape(12.dp),
                    modifier = Modifier
                        .graphicsLayer {
                            if (isNarrating) {
                                scaleX = scale
                                scaleY = scale
                            }
                        }
                        .clickable {
                            vibrate(context)
                            if (isNarrating) {
                                tts.stop()
                            }
                            vm.toggleNarration(
                                getFrames = {
                                    // Visual/Tactile feedback that a capture is starting
                                    vibrate(context)

                                    val newFrames = mutableListOf<Bitmap>()
                                    val executor = ContextCompat.getMainExecutor(context)
                                    for (i in 1..2) {
                                        val deferred = kotlinx.coroutines.CompletableDeferred<Bitmap>()
                                        imageCapture?.takePicture(executor, object : ImageCapture.OnImageCapturedCallback() {
                                            override fun onCaptureSuccess(image: ImageProxy) {
                                                deferred.complete(image.toBitmap()); image.close()
                                            }
                                            override fun onError(exception: ImageCaptureException) {
                                                deferred.completeExceptionally(exception)
                                            }
                                        })
                                        try { newFrames.add(deferred.await()); delay(400) } catch (_: Exception) {}
                                    }
                                    newFrames
                                },
                                onSpeak = { tts.speak(it, TextToSpeech.QUEUE_FLUSH, null, null) }
                            )
                        }
                ) {
                    Row(Modifier.padding(horizontal = 12.dp, vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                        Icon(
                            imageVector = if (isScanning) Icons.Default.Sync else if (isNarrating) Icons.Default.VolumeUp else Icons.Default.VolumeOff,
                            contentDescription = null,
                            tint = if (isNarrating) tc.onAccent else Color.White,
                            modifier = Modifier.size(16.dp).graphicsLayer {
                                if (isScanning) rotationZ = alpha * 360f
                            }
                        )
                        Spacer(Modifier.width(8.dp))
                        Text(
                            text = if (isScanning) "SCANNING…" else if (isNarrating) "NARRATOR ON" else "NARRATOR OFF",
                            color = if (isNarrating) tc.onAccent else Color.White,
                            fontWeight = FontWeight.Bold,
                            fontSize = 12.sp
                        )
                    }
                }
            }

            // Bottom Section
            val liveSessionFrames = remember { mutableStateListOf<Bitmap>() }
            var isWatching by remember { mutableStateOf(false) }

            Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                if (state is UiState.Done) {
                    LiveResultSection(
                        vm = vm, 
                        result = (state as UiState.Done).result, 
                        question = question, 
                        tc = tc, 
                        context = context, 
                        tts = tts, 
                        onClear = {
                            tts.stop()
                            liveSessionFrames.clear()
                            question = ""
                            vm.resetLive()
                        },
                        onWatchMore = {
                            tts.stop()
                            vm.resetLiveStateOnly() 
                        }
                    )
                }

                if (state is UiState.Loading) {
                    LiveLoadingSection(state as UiState.Loading, progress, tc, vm)
                } else if (state !is UiState.Done) {
                    Surface(color = tc.card.copy(alpha = 0.95f), shape = RoundedCornerShape(24.dp), border = BorderStroke(1.dp, tc.accent.copy(alpha = 0.5f))) {
                        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                            
                            OutlinedTextField(
                                value = question,
                                onValueChange = { question = it },
                                modifier = Modifier.fillMaxWidth(),
                                placeholder = { Text("What am I looking at?", color = tc.sub) },
                                textStyle = androidx.compose.ui.text.TextStyle(color = tc.text),
                                trailingIcon = {
                    IconButton(onClick = {
                    vibrate(context)
                    val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                    }
                    speechLauncher.launch(intent)
                }) { Icon(Icons.Filled.Mic, null, tint = tc.accent) }
                                },
                                colors = OutlinedTextFieldDefaults.colors(
                                    focusedTextColor = tc.text, 
                                    unfocusedTextColor = tc.text,
                                    focusedBorderColor = tc.accent,
                                    unfocusedBorderColor = tc.sub.copy(alpha = 0.3f)
                                )
                            )

                            Button(
                                onClick = {
                                    vibrate(context)
                                    isWatching = true
                                    focusManager.clearFocus()
                                    tts.stop()
                                    vm.setLiveStatus("Capturing Live Feed…", 0.1f)

                                    val job = scope.launch(Dispatchers.IO) {
                                        val executor = ContextCompat.getMainExecutor(context)
                                        // Capture a few frames for context
                                        val newFrames = mutableListOf<Bitmap>()
                                        var capturedCount = 0
                                        while (capturedCount < 3) {
                                            if (!isActive) break
                                            val deferred = kotlinx.coroutines.CompletableDeferred<Bitmap>()
                                            imageCapture?.takePicture(executor, object : ImageCapture.OnImageCapturedCallback() {
                                                override fun onCaptureSuccess(image: ImageProxy) {
                                                    deferred.complete(image.toBitmap())
                                                    image.close()
                                                }
                                                override fun onError(exception: ImageCaptureException) {
                                                    deferred.completeExceptionally(exception)
                                                }
                                            })
                                            try {
                                                val bmp = deferred.await()
                                                newFrames.add(bmp)
                                                capturedCount++
                                                delay(500)
                                            } catch (_: Exception) { break }
                                        }
                                        
                                        if (isActive && newFrames.isNotEmpty()) {
                                            liveSessionFrames.addAll(newFrames)
                                            vm.runOnLiveVideo(liveSessionFrames.toList(), question, vm.currentChatHistory.value)
                                        }
                                        isWatching = false
                                    }
                                    vm.setLiveCaptureJob(job)
                                },
                                modifier = Modifier.fillMaxWidth().height(56.dp),
                                enabled = question.isNotBlank() && imageCapture != null && !isWatching,
                                shape = RoundedCornerShape(16.dp),
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = tc.accent,
                                    contentColor = tc.onAccent,
                                    disabledContainerColor = tc.accent.copy(alpha = 0.2f),
                                    disabledContentColor = tc.onAccent.copy(alpha = 0.5f)
                                )
                            ) {
                                Icon(Icons.Default.PlayArrow, null)
                                Spacer(Modifier.width(12.dp))
                                Text(if (liveSessionFrames.isEmpty()) "Start Watching" else "Watch More", fontWeight = FontWeight.Bold)
                            }

                            if (liveSessionFrames.isNotEmpty() && !isWatching) {
                                OutlinedButton(
                                    onClick = {
                                        vibrate(context)
                                        liveSessionFrames.clear()
                                        question = ""
                                        vm.resetLive()
                                    },
                                    modifier = Modifier.fillMaxWidth(),
                                    border = BorderStroke(1.dp, tc.sub)
                                ) {
                                    Text("Clear All", color = tc.text)
                                }
                            }
                        }
                    }
                }
                Spacer(Modifier.height(12.dp)) // Extra safety margin at the bottom
            }
        }
    }
}

// ── Sub-Sections ─────────────────────────────────────────────────────────────

/**
 * UI section for selecting media (Gallery or Camera).
 */
@Composable
fun MediaPickerSection(tc: ThemeColors, imagePicker: androidx.activity.result.ActivityResultLauncher<Array<String>>, context: Context, cameraCaptureLauncher: androidx.activity.result.ActivityResultLauncher<Uri>, cameraVideoLauncher: androidx.activity.result.ActivityResultLauncher<Uri>, onCameraUri: (Uri) -> Unit, onVideoUri: (Uri) -> Unit) {
    val haptic = LocalHapticFeedback.current
    Surface(
        color = tc.card,
        shape = RoundedCornerShape(20.dp),
        border = BorderStroke(1.dp, tc.sub.copy(alpha = 0.2f)),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(Modifier.padding(32.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(Icons.Default.Add, null, modifier = Modifier.size(64.dp), tint = tc.accent)
            Spacer(Modifier.height(16.dp))
            Text("Select media to start", color = tc.text, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(24.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = {
                    vibrate(context)
                    imagePicker.launch(arrayOf("image/*", "video/*"))
                }, modifier = Modifier.weight(1f), colors = ButtonDefaults.buttonColors(containerColor = tc.accent, contentColor = tc.onAccent)) {
                    Text("Gallery")
                }
                var showCameraChoice by remember { mutableStateOf(false) }
                OutlinedButton(onClick = {
                    vibrate(context)
                    showCameraChoice = true
                }, modifier = Modifier.weight(1f), border = BorderStroke(1.dp, tc.sub)) {
                    Text("Camera", color = tc.text)
                }
                if (showCameraChoice) CameraChoiceDialog(tc, context, cameraCaptureLauncher, cameraVideoLauncher, onCameraUri, onVideoUri) { showCameraChoice = false }
            }
        }
    }
}

/**
 * Video player section with custom playback controls.
 */
@Composable
fun VideoPlayerSection(uri: Uri, isPlaying: Boolean, onPlayingChange: (Boolean) -> Unit, tc: ThemeColors, imagePicker: androidx.activity.result.ActivityResultLauncher<Array<String>>, context: Context, cameraCaptureLauncher: androidx.activity.result.ActivityResultLauncher<Uri>, cameraVideoLauncher: androidx.activity.result.ActivityResultLauncher<Uri>, onCameraUri: (Uri) -> Unit, onVideoUri: (Uri) -> Unit) {
    val haptic = LocalHapticFeedback.current
    var videoViewRef by remember { mutableStateOf<VideoView?>(null) }
    var currentPos by remember { mutableStateOf(0f) }
    var duration by remember { mutableStateOf(1f) }

    LaunchedEffect(isPlaying, videoViewRef) {
        while (isPlaying && videoViewRef != null) {
            currentPos = videoViewRef!!.currentPosition.toFloat()
            duration = videoViewRef!!.duration.toFloat().coerceAtLeast(1f)
            delay(500)
        }
    }

    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Box(Modifier.fillMaxWidth().heightIn(min = 240.dp, max = 450.dp).clip(RoundedCornerShape(16.dp)).background(Color.Black)) {
            AndroidView(
                factory = { ctx ->
                    VideoView(ctx).also { videoView ->
                        videoView.setVideoURI(uri)
                        videoView.tag = uri
                        videoView.setOnPreparedListener { mp ->
                            mp.isLooping = true
                            duration = videoView.duration.toFloat().coerceAtLeast(1f)
                            if (isPlaying) videoView.start()
                        }
                        videoViewRef = videoView
                    }
                },
                update = { videoView ->
                    if (videoView.tag != uri) {
                        videoView.setVideoURI(uri)
                        videoView.tag = uri
                        if (isPlaying) videoView.start()
                    }
                    if (isPlaying && !videoView.isPlaying) videoView.start()
                    else if (!isPlaying && videoView.isPlaying) videoView.pause()
                },
                modifier = Modifier.fillMaxSize()
            )
        }

        // Custom Media Controller
        Surface(
            color = tc.card2,
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(Modifier.padding(8.dp)) {
                Slider(
                    value = currentPos,
                    onValueChange = {
                        vibrate(context)
                        currentPos = it
                        videoViewRef?.seekTo(it.toInt())
                    },
                    valueRange = 0f..duration,
                    colors = SliderDefaults.colors(thumbColor = tc.accent, activeTrackColor = tc.accent)
                )
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text(formatTime(currentPos.toInt()), fontSize = 10.sp, color = tc.sub)
                    Text(formatTime(duration.toInt()), fontSize = 10.sp, color = tc.sub)
                }
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = {
                vibrate(context)
                onPlayingChange(!isPlaying)
            }, modifier = Modifier.weight(1f), colors = ButtonDefaults.buttonColors(containerColor = tc.accent, contentColor = tc.onAccent)) {
                Text(if (isPlaying) "⏸ Stop" else "▶ Play")
            }
            OutlinedButton(onClick = {
                vibrate(context)
                imagePicker.launch(arrayOf("image/*", "video/*"))
            }, modifier = Modifier.weight(1f), border = BorderStroke(1.dp, tc.sub)) {
                Text("Gallery", color = tc.text)
            }
            var showCameraChoice by remember { mutableStateOf(false) }
            OutlinedButton(onClick = {
                vibrate(context)
                showCameraChoice = true
            }, modifier = Modifier.weight(1f), border = BorderStroke(1.dp, tc.sub)) {
                Text("Camera", color = tc.text)
            }
            if (showCameraChoice) CameraChoiceDialog(tc, context, cameraCaptureLauncher, cameraVideoLauncher, onCameraUri, onVideoUri) { showCameraChoice = false }
        }
    }
}

/**
 * Preview section for selected images.
 */
@Composable
fun ImagePreviewSection(uri: Uri, tc: ThemeColors, context: Context, imagePicker: androidx.activity.result.ActivityResultLauncher<Array<String>>, cameraCaptureLauncher: androidx.activity.result.ActivityResultLauncher<Uri>, cameraVideoLauncher: androidx.activity.result.ActivityResultLauncher<Uri>, onCameraUri: (Uri) -> Unit, onVideoUri: (Uri) -> Unit) {
    val haptic = LocalHapticFeedback.current
    val bitmap = remember(uri) {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, uri)) { d, _, _ ->
                    d.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                }
            } else {
                @Suppress("DEPRECATION")
                MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
            }
        } catch (_: Exception) { null }
    }
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Box(Modifier.fillMaxWidth().heightIn(max = 400.dp).clip(RoundedCornerShape(16.dp)).background(tc.card), contentAlignment = Alignment.Center) {
            if (bitmap != null) Image(
                bitmap.asImageBitmap(),
                contentDescription = "Selected image preview",
                contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize()
            )
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = {
                vibrate(context)
                imagePicker.launch(arrayOf("image/*", "video/*"))
            }, modifier = Modifier.weight(1f), colors = ButtonDefaults.buttonColors(containerColor = tc.accent, contentColor = tc.onAccent)) {
                Text("Gallery")
            }
            var showChoice by remember { mutableStateOf(false) }
            OutlinedButton(onClick = {
                vibrate(context)
                showChoice = true
            }, modifier = Modifier.weight(1f), border = BorderStroke(1.dp, tc.sub)) {
                Text("Camera", color = tc.text)
            }
            if (showChoice) CameraChoiceDialog(tc, context, cameraCaptureLauncher, cameraVideoLauncher, onCameraUri, onVideoUri) { showChoice = false }
        }
    }
}

/**
 * Loading state display for home inference.
 */
@Composable
fun HomeLoadingSection(loading: UiState.Loading, progress: Float, tc: ThemeColors, vm: InferenceViewModel, focusManager: androidx.compose.ui.focus.FocusManager) {
    val context = LocalContext.current
    val partial by vm.homePartialResult.collectAsState()
    val haptic = LocalHapticFeedback.current

    Surface(
        color = tc.card,
        shape = RoundedCornerShape(16.dp),
        modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite }
    ) {
        Column(Modifier.padding(20.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(loading.message, color = tc.text, fontSize = 14.sp)
                Text("${(progress * 100).toInt()}%", color = tc.accent, fontWeight = FontWeight.Bold)
            }
            Spacer(Modifier.height(12.dp))
            LinearProgressIndicator(progress = progress, modifier = Modifier.fillMaxWidth().height(8.dp).clip(CircleShape), color = tc.accent, trackColor = tc.card2)
            
            if (partial.isNotEmpty()) {
                Spacer(Modifier.height(16.dp))
                Text(
                    text = partial,
                    color = tc.text,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Medium,
                    modifier = Modifier.fillMaxWidth().background(tc.card2.copy(alpha = 0.5f), RoundedCornerShape(8.dp)).padding(12.dp)
                )
            }

            Spacer(Modifier.height(16.dp))
            TextButton(onClick = {
                vibrate(context)
                vm.cancelHomeInference(); focusManager.clearFocus()
            }, modifier = Modifier.align(Alignment.End)) {
                Text("Abort", color = Color.Red, fontWeight = FontWeight.Bold)
            }
        }
    }
}

/**
 * Loading state display for live inference.
 */
@Composable
fun LiveLoadingSection(loading: UiState.Loading, progress: Float, tc: ThemeColors, vm: InferenceViewModel) {
    val context = LocalContext.current
    val partial by vm.livePartialResult.collectAsState()
    val haptic = LocalHapticFeedback.current

    Surface(
        color = tc.card.copy(alpha = 0.9f),
        shape = RoundedCornerShape(16.dp),
        modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite }
    ) {
        Column(Modifier.padding(20.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(loading.message, color = tc.text, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                Text("${(progress * 100).toInt()}%", color = tc.accent, fontWeight = FontWeight.Bold)
            }
            Spacer(Modifier.height(12.dp))
            LinearProgressIndicator(progress = progress, modifier = Modifier.fillMaxWidth().height(8.dp).clip(CircleShape), color = tc.accent, trackColor = tc.card2)
            
            if (partial.isNotEmpty()) {
                Spacer(Modifier.height(16.dp))
                Text(
                    text = partial,
                    color = tc.text,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Medium,
                    modifier = Modifier.fillMaxWidth().background(tc.card2.copy(alpha = 0.5f), RoundedCornerShape(8.dp)).padding(12.dp)
                )
            }

            Spacer(Modifier.height(16.dp))
            TextButton(
                onClick = {
                    vibrate(context)
                    vm.resetLive()
                },
                modifier = Modifier.align(Alignment.End)
            ) {
                Text("Cancel Analysis", color = Color.Red, fontWeight = FontWeight.Bold)
            }
        }
    }
}

/**
 * Displays the result of an analysis on the home screen.
 */
@OptIn(ExperimentalComposeUiApi::class)
@Composable
fun HomeResultSection(vm: InferenceViewModel, result: InferenceResult.Success, question: String, tc: ThemeColors, context: Context, tts: TextToSpeech, onResetQuestion: () -> Unit) {
    val videoUri by vm.selectedVideo.collectAsState()
    val imageUri by vm.selectedImage.collectAsState()
    val lastStats by vm.homeLastStats.collectAsState()
    val chatHistory by vm.currentChatHistory.collectAsState()

    Surface(
        color = tc.card,
        shape = RoundedCornerShape(20.dp),
        border = BorderStroke(1.5.dp, tc.accent.copy(alpha = 0.2f)),
        modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite }
    ) {
        Column(Modifier.padding(20.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Analysis", color = tc.sub, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                    Spacer(Modifier.width(8.dp))
                    Surface(color = tc.accent.copy(alpha = 0.1f), shape = CircleShape) {
                        Text("Completed", color = tc.accent, fontSize = 10.sp, modifier = Modifier.padding(horizontal = 8.dp, vertical = 2.dp))
                    }
                }
                Row {
                    IconButton(onClick = {
                        vibrate(context)
                        val shareUri = videoUri ?: imageUri
                        val intent = Intent(Intent.ACTION_SEND).apply {
                            type = if (shareUri != null) context.contentResolver.getType(shareUri) ?: "*/*" else "text/plain"
                            val historyText = chatHistory.joinToString("\n") { "Q: ${it.first}\nA: ${it.second}" }
                            if (shareUri != null) {
                                putExtra(Intent.EXTRA_STREAM, shareUri)
                                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                            }
                            putExtra(Intent.EXTRA_TEXT, "$historyText\n\nShared from Mozix.ai")
                        }
                        context.startActivity(Intent.createChooser(intent, "Share"))
                    }) { Icon(Icons.Default.Share, null, tint = tc.sub) }
                }
            }

            // Chat History display
            Column(Modifier.padding(vertical = 12.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                chatHistory.forEach { (q, a) ->
                    Column {
                        Text("Q: $q", fontSize = 14.sp, color = tc.sub, fontWeight = FontWeight.Medium)
                        Text(a, fontSize = 19.sp, fontWeight = FontWeight.Bold, color = if (a == result.answer) tc.accent else tc.text)
                    }
                }
            }

            // ── Stats Section ──
            Spacer(Modifier.height(8.dp))
            Divider(color = tc.sub.copy(alpha = 0.15f))
            Spacer(Modifier.height(12.dp))
            
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Column {
                    Text("Time", color = tc.sub, fontSize = 10.sp)
                    Text("${result.elapsedMs}ms", color = tc.text, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                }
                lastStats?.let {
                    Column {
                        Text("Extracted", color = tc.sub, fontSize = 10.sp)
                        Text("${it.totalFrames}", color = tc.text, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                    }
                }
                Column {
                    Text("Top K", color = tc.sub, fontSize = 10.sp)
                    Text("${result.selectedFrames.size}", color = tc.text, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                }
                lastStats?.let {
                    Column {
                        Text("Tokens", color = tc.sub, fontSize = 10.sp)
                        Text("${it.tokensGenerated}", color = tc.text, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                    }
                }
            }

            Spacer(Modifier.height(24.dp))
            
            val sentences = remember(result.answer) { 
                result.answer.split(Regex("(?<=[.!?])\\s+")).filter { it.isNotBlank() } 
            }
            var currentSentenceIndex by remember(result.answer) { mutableStateOf(0) }
            var ttsPlaying by remember(result.answer) { mutableStateOf(true) }

            LaunchedEffect(ttsPlaying, currentSentenceIndex) {
                if (ttsPlaying && currentSentenceIndex < sentences.size) {
                    tts.speak(sentences[currentSentenceIndex], TextToSpeech.QUEUE_FLUSH, null, "utterance_${currentSentenceIndex}")
                    while (tts.isSpeaking && ttsPlaying) {
                        delay(200)
                    }
                    if (ttsPlaying) {
                        if (currentSentenceIndex < sentences.size - 1) {
                            currentSentenceIndex++
                        } else {
                            ttsPlaying = false
                            currentSentenceIndex = 0
                        }
                    }
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = {
                    vibrate(context)
                    if (ttsPlaying) { tts.stop(); ttsPlaying = false }
                    else { ttsPlaying = true }
                }, modifier = Modifier.fillMaxWidth(), colors = ButtonDefaults.buttonColors(containerColor = if (ttsPlaying) Color(0xFFFFA500) else tc.accent, contentColor = tc.onAccent)) {
                    Icon(if (ttsPlaying) Icons.Filled.Pause else Icons.Default.PlayArrow, null)
                    Spacer(Modifier.width(8.dp))
                    Text(if (ttsPlaying) "Pause" else if (currentSentenceIndex > 0) "Resume" else "Listen")
                }
            }
            
            Spacer(Modifier.height(12.dp))
            OutlinedButton(onClick = {
                vibrate(context)
                onResetQuestion()
            }, modifier = Modifier.fillMaxWidth()) {
                Text("💬 New Question", color = tc.text)
            }

            // ── Selected Key Frames (Full View) ──
            if (result.selectedFrames.isNotEmpty()) {
                Spacer(Modifier.height(24.dp))
                Text("Key Frames Analysed", color = tc.sub, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                Spacer(Modifier.height(8.dp))
                
                val frames = remember(videoUri, imageUri, result.selectedFrames) {
                    if (videoUri != null) {
                        extractPreviewFrames(context, videoUri!!, result.selectedFrames, result.totalFrames)
                    } else if (imageUri != null) {
                        // For images, we just show the image itself
                        listOfNotNull(try {
                             if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, imageUri!!)) { d, _, _ -> d.allocator = ImageDecoder.ALLOCATOR_SOFTWARE }
                             else MediaStore.Images.Media.getBitmap(context.contentResolver, imageUri!!)
                        } catch(_: Exception) { null })
                    } else emptyList()
                }

                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    frames.forEach { bmp ->
                        Image(
                            bitmap = bmp.asImageBitmap(),
                            contentDescription = "Selected frame",
                            contentScale = ContentScale.FillWidth,
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(12.dp))
                                .background(Color.Black)
                        )
                    }
                }
            }
        }
    }
}

/**
 * Displays the result of a live analysis.
 */
@OptIn(ExperimentalComposeUiApi::class)
@Composable
fun LiveResultSection(vm: InferenceViewModel, result: InferenceResult.Success, question: String, tc: ThemeColors, context: Context, tts: TextToSpeech, onClear: () -> Unit, onWatchMore: () -> Unit) {
    val lastStats by vm.liveLastStats.collectAsState()
    val chatHistory by vm.currentChatHistory.collectAsState()
    val haptic = LocalHapticFeedback.current

    Surface(
        color = tc.card,
        shape = RoundedCornerShape(20.dp),
        border = BorderStroke(1.5.dp, tc.accent.copy(alpha = 0.2f)),
        modifier = Modifier.semantics { liveRegion = LiveRegionMode.Polite }
    ) {
        Column(Modifier.padding(20.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("Conversation", color = tc.sub, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                }
                Row {
                    IconButton(onClick = {
                        vibrate(context)
                        val intent = Intent(Intent.ACTION_SEND).apply {
                            type = "text/plain"
                            val histText = chatHistory.joinToString("\n") { "Q: ${it.first}\nA: ${it.second}\n" }
                            putExtra(Intent.EXTRA_TEXT, "History:\n$histText\nShared from AI Assistant")
                        }
                        context.startActivity(Intent.createChooser(intent, "Share"))
                    }) { Icon(Icons.Default.Share, null, tint = tc.sub) }
                }
            }

            // Chat History
            Column(
                modifier = Modifier
                    .padding(vertical = 8.dp)
                    .heightIn(max = 300.dp)
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                chatHistory.forEach { pair ->
                    Column {
                        Text("Q: ${pair.first}", fontSize = 14.sp, color = tc.sub)
                        Text(pair.second, fontSize = 18.sp, fontWeight = FontWeight.Bold, color = if (pair == chatHistory.last()) tc.accent else tc.text)
                    }
                }
            }

            // ── Stats Section ──
            Spacer(Modifier.height(16.dp))
            Divider(color = tc.sub.copy(alpha = 0.2f))
            Spacer(Modifier.height(12.dp))
            
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Column {
                    Text("Time", color = tc.sub, fontSize = 11.sp)
                    Text("${result.elapsedMs}ms", color = tc.text, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                }
                lastStats?.let {
                    Column {
                        Text("Tokens", color = tc.sub, fontSize = 11.sp)
                        Text("${it.tokensGenerated}", color = tc.text, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                    }
                }
                Column {
                    Text("Frames", color = tc.sub, fontSize = 11.sp)
                    Text("${result.selectedFrames.size}", color = tc.text, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                }
            }

            Spacer(Modifier.height(24.dp))
            
            val sentences = remember(result.answer) { 
                result.answer.split(Regex("(?<=[.!?])\\s+")).filter { it.isNotBlank() } 
            }
            var currentSentenceIndex by remember(result.answer) { mutableStateOf(0) }
            var ttsPlaying by remember(result.answer) { mutableStateOf(true) }

            LaunchedEffect(ttsPlaying, currentSentenceIndex) {
                if (ttsPlaying && currentSentenceIndex < sentences.size) {
                    tts.speak(sentences[currentSentenceIndex], TextToSpeech.QUEUE_FLUSH, null, "live_${currentSentenceIndex}")
                    while (tts.isSpeaking && ttsPlaying) {
                        delay(200)
                    }
                    if (ttsPlaying) {
                        if (currentSentenceIndex < sentences.size - 1) {
                            currentSentenceIndex++
                        } else {
                            ttsPlaying = false
                            currentSentenceIndex = 0
                        }
                    }
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = {
                    vibrate(context)
                    if (ttsPlaying) { tts.stop(); ttsPlaying = false }
                    else { ttsPlaying = true }
                }, modifier = Modifier.fillMaxWidth(), colors = ButtonDefaults.buttonColors(containerColor = if (ttsPlaying) Color(0xFFFFA500) else tc.accent, contentColor = tc.onAccent)) {
                    Icon(if (ttsPlaying) Icons.Filled.Pause else Icons.Default.PlayArrow, null)
                    Spacer(Modifier.width(8.dp))
                    Text(if (ttsPlaying) "Pause" else if (currentSentenceIndex > 0) "Resume" else "Listen")
                }
            }
            
            Spacer(Modifier.height(16.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedButton(onClick = {
                    vibrate(context)
                    onWatchMore()
                }, modifier = Modifier.weight(1f), border = BorderStroke(1.dp, tc.sub)) {
                    Icon(Icons.Default.Add, null, tint = tc.text)
                    Spacer(Modifier.width(8.dp))
                    Text("Watch More", color = tc.text)
                }
                Button(onClick = {
                    vibrate(context)
                    onClear()
                }, modifier = Modifier.weight(1f), colors = ButtonDefaults.buttonColors(containerColor = tc.accent, contentColor = tc.onAccent)) {
                    Icon(Icons.Default.Refresh, null)
                    Spacer(Modifier.width(8.dp))
                    Text("Clear All")
                }
            }
        }
    }
}

// ── History Screen ────────────────────────────────────────────────────────────

/**
 * Screen displaying the history of previous analyses and bookmarked entries.
 */
@Composable
fun HistoryScreen(vm: InferenceViewModel, tc: ThemeColors, onNavigateToHome: () -> Unit) {
    val history by vm.history.collectAsState()
    val bookmarks by vm.bookmarks.collectAsState()
    val context = LocalContext.current
    var showBookmarksOnly by remember { mutableStateOf(false) }

    Column(Modifier.fillMaxSize().padding(20.dp)) {
        val haptic = LocalHapticFeedback.current
        Text("Activity", fontSize = 28.sp, fontWeight = FontWeight.Bold, color = tc.text)
        Spacer(Modifier.height(16.dp))
        
        TabRow(
            selectedTabIndex = if (showBookmarksOnly) 1 else 0,
            containerColor = Color.Transparent,
            contentColor = tc.accent,
            indicator = { tabPositions ->
                if ((if (showBookmarksOnly) 1 else 0) < tabPositions.size) {
                    TabRowDefaults.Indicator(
                        modifier = Modifier.tabIndicatorOffset(tabPositions[if (showBookmarksOnly) 1 else 0]),
                        color = tc.accent
                    )
                }
            },
            divider = {}
        ) {
            Tab(selected = !showBookmarksOnly, onClick = {
                vibrate(context)
                showBookmarksOnly = false
            }, text = { Text("Recent", color = if (!showBookmarksOnly) tc.accent else tc.sub) })
            Tab(selected = showBookmarksOnly, onClick = {
                vibrate(context)
                showBookmarksOnly = true
            }, text = { Text("Starred", color = if (showBookmarksOnly) tc.accent else tc.sub) })
        }

        Spacer(Modifier.height(12.dp))
        val list = if (showBookmarksOnly) bookmarks else history.reversed()

        if (list.isEmpty()) {
            Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
                Text("No entries found", color = tc.sub)
            }
        } else {
            LazyColumn(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                items(list) { entry ->
                    HistoryCard(entry, tc, vm, context, onDelete = { vm.deleteHistoryEntry(it) }, onClick = {
                        vm.restoreEntry(it)
                        onNavigateToHome()
                    })
                }
            }
        }

        
        if (!showBookmarksOnly && list.isNotEmpty()) {
            TextButton(onClick = {
                vibrate(context)
                vm.clearHistory()
            }, modifier = Modifier.align(Alignment.CenterHorizontally)) {
                Text("Clear History", color = Color.Red)
            }
        }
    }
}

// ── Community Screen ─────────────────────────────────────────────────────────

/**
 * Data class representing a volunteer in the community.
 */
data class Volunteer(
    val name: String,
    val bio: String,
    val rating: Float,
    val specialties: List<String>,
    val isOpen: Boolean,
    val location: String,
    val distanceKm: Double
)

/**
 * Data class for a category of community assistance.
 */
data class CommunityCategory(
    val title: String,
    val icon: androidx.compose.ui.graphics.vector.ImageVector,
    val description: String,
    val color: Color
)

/**
 * Community screen for connecting users with volunteers.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CommunityScreen(tc: ThemeColors) {
    val haptic = LocalHapticFeedback.current
    val categories = remember {
        listOf(
            CommunityCategory("Shopping & Essentials", Icons.Default.ShoppingCart, "Help with grocery shopping or buying daily resources.", Color(0xFF4CAF50)),
            CommunityCategory("Medical Assistance", Icons.Default.Favorite, "Support for clinic visits, medicine reading, or health checks.", Color(0xFFE91E63)),
            CommunityCategory("Study & Coaching", Icons.Default.Edit, "Assistance with reading, learning, or academic coaching.", Color(0xFF2196F3)),
            CommunityCategory("Social & Get Together", Icons.Default.Groups, "Connect for dinners, games, or simply spending time together.", Color(0xFFFF9800)),
            CommunityCategory("Trips & Travel", Icons.Default.LocationOn, "Safe guidance and companionship for outdoor trips or walks.", Color(0xFF9C27B0))
        )
    }

    var selectedVolunteer by remember { mutableStateOf<Volunteer?>(null) }
    var selectedCategory by remember { mutableStateOf<String?>(null) }

    val mockVolunteers = remember {
        listOf(
            Volunteer("Alice Smith", "Dedicated helper with 5 years experience.", 4.9f, listOf("Shopping & Essentials", "Medical Assistance"), true, "Downtown", 2.4),
            Volunteer("Bob Johnson", "Retired teacher, happy to help with study.", 4.7f, listOf("Study & Coaching", "Social & Get Together"), true, "Uptown", 15.2),
            Volunteer("Charlie Brown", "Avid walker and traveler.", 4.8f, listOf("Trips & Travel"), true, "East Side", 8.1),
            Volunteer("Diana Prince", "Helping hands for everyone.", 4.6f, listOf("Shopping & Essentials"), true, "West Side", 25.0),
            Volunteer("Ethan Hunt", "Quick responder for medical emergencies.", 5.0f, listOf("Medical Assistance"), true, "Suburbs", 42.5),
            Volunteer("Fiona Glenanne", "Ex-intelligence, very observant and helpful.", 4.8f, listOf("Trips & Travel", "Social & Get Together"), true, "South Beach", 5.0),
            Volunteer("George Bluth", "I'll help you with your groceries and errands.", 4.2f, listOf("Shopping & Essentials"), true, "Orange County", 12.3),
            Volunteer("Helen Parr", "Flexible and fast helper.", 5.0f, listOf("Shopping & Essentials", "Medical Assistance", "Trips & Travel"), true, "Metroville", 1.5),
            Volunteer("Ian Malcolm", "Life finds a way, and so does my help with study.", 4.5f, listOf("Study & Coaching"), true, "Jurassic Park", 18.0),
            Volunteer("Jane Doe", "Local guide for social walks.", 4.7f, listOf("Social & Get Together", "Trips & Travel"), true, "Greenwich", 3.2)
        )
    }

    Column(Modifier.fillMaxSize().padding(20.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(20.dp)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
            Text("Community", fontSize = 28.sp, fontWeight = FontWeight.Bold, color = tc.text)
        }
        
        // Category filtering cards
        Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            // "All" Card
            CategoryCard(
                title = "All",
                icon = Icons.Default.Apps,
                color = tc.accent,
                isSelected = selectedCategory == null,
                tc = tc,
                onClick = { selectedCategory = null }
            )
            categories.forEach { cat ->
                CategoryCard(
                    title = cat.title,
                    icon = cat.icon,
                    color = cat.color,
                    isSelected = selectedCategory == cat.title,
                    tc = tc,
                    onClick = { selectedCategory = cat.title }
                )
            }
        }

        if (selectedCategory == null) {
            Surface(
                color = tc.accent.copy(alpha = 0.1f),
                shape = RoundedCornerShape(20.dp),
                border = BorderStroke(1.5.dp, tc.accent),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(Modifier.padding(20.dp)) {
                    Text("Nearby Assistance", fontWeight = FontWeight.Bold, color = tc.accent, fontSize = 18.sp)
                    Text("Showing trusted volunteers within your 20km range.", color = tc.text, fontSize = 14.sp)
                }
            }

            Text("Volunteers Near You", fontSize = 20.sp, fontWeight = FontWeight.Bold, color = tc.text)
            val nearbyVolunteers = mockVolunteers.filter { it.isOpen && it.distanceKm <= 20.0 }

            if (nearbyVolunteers.isEmpty()) {
                Box(Modifier.fillMaxWidth().height(100.dp).background(tc.card2, RoundedCornerShape(16.dp)), contentAlignment = Alignment.Center) {
                    Text("No volunteers found within 20km", color = tc.sub)
                }
            } else {
                nearbyVolunteers.forEach { volunteer ->
                    VolunteerCard(volunteer, tc) { selectedVolunteer = volunteer }
                }
            }
        } else {
            Text("Volunteers in $selectedCategory", fontSize = 20.sp, fontWeight = FontWeight.Bold, color = tc.text)
            val filtered = mockVolunteers.filter { it.specialties.contains(selectedCategory) }
            if (filtered.isEmpty()) {
                Box(Modifier.fillMaxWidth().height(100.dp).background(tc.card2, RoundedCornerShape(16.dp)), contentAlignment = Alignment.Center) {
                    Text("No volunteers found in this category", color = tc.sub)
                }
            } else {
                filtered.forEach { volunteer ->
                    VolunteerCard(volunteer, tc) { selectedVolunteer = volunteer }
                }
            }
        }
        
        Spacer(Modifier.height(20.dp))
    }

    if (selectedVolunteer != null) {
        VolunteerDetailDialog(selectedVolunteer!!, tc) { selectedVolunteer = null }
    }
}

/**
 * A selectable card for community categories.
 */
@Composable
fun CategoryCard(title: String, icon: androidx.compose.ui.graphics.vector.ImageVector, color: Color, isSelected: Boolean, tc: ThemeColors, onClick: () -> Unit) {
    val context = LocalContext.current
    val haptic = LocalHapticFeedback.current
    Surface(
        onClick = {
            vibrate(context)
            onClick()
        },
        color = if (isSelected) tc.accent.copy(alpha = 0.15f) else tc.card,
        shape = RoundedCornerShape(16.dp),
        border = BorderStroke(1.5.dp, if (isSelected) tc.accent else tc.sub.copy(alpha = 0.1f)),
        modifier = Modifier.width(100.dp).height(110.dp)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Surface(color = color.copy(alpha = 0.15f), shape = CircleShape, modifier = Modifier.size(40.dp)) {
                Icon(icon, null, tint = color, modifier = Modifier.padding(10.dp))
            }
            Spacer(Modifier.height(8.dp))
            Text(title, fontSize = 11.sp, fontWeight = FontWeight.Bold, color = if (isSelected) tc.accent else tc.text, textAlign = TextAlign.Center, maxLines = 2, lineHeight = 13.sp)
        }
    }
}

/**
 * A single item in the community category list.
 */
@Composable
fun CommunityItem(cat: CommunityCategory, tc: ThemeColors, isCheckable: Boolean = false, onClick: () -> Unit) {
    val context = LocalContext.current
    var checked by remember { mutableStateOf(false) }
    val haptic = LocalHapticFeedback.current
    Surface(
        color = tc.card,
        shape = RoundedCornerShape(16.dp),
        border = BorderStroke(1.dp, tc.sub.copy(alpha = 0.2f)),
        onClick = { 
            vibrate(context)
            if (isCheckable) checked = !checked
            onClick() 
        },
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)
    ) {
        Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Surface(color = cat.color.copy(alpha = 0.15f), shape = CircleShape, modifier = Modifier.size(48.dp)) {
                Icon(cat.icon, null, tint = cat.color, modifier = Modifier.padding(12.dp))
            }
            Spacer(Modifier.width(16.dp))
            Column(Modifier.weight(1f)) {
                Text(cat.title, fontWeight = FontWeight.Bold, color = tc.text)
                Text(cat.description, color = tc.sub, fontSize = 12.sp)
            }
            if (isCheckable) {
                Checkbox(checked = checked, onCheckedChange = { checked = it; onClick() })
            } else {
                Icon(Icons.Default.KeyboardArrowRight, null, tint = tc.sub)
            }
        }
    }
}

/**
 * Displays a summary card for a volunteer.
 */
@Composable
fun VolunteerCard(v: Volunteer, tc: ThemeColors, onClick: () -> Unit) {
    val context = LocalContext.current
    val haptic = LocalHapticFeedback.current
    Surface(
        color = tc.card,
        shape = RoundedCornerShape(16.dp),
        border = BorderStroke(1.dp, tc.sub.copy(alpha = 0.2f)),
        onClick = {
            vibrate(context)
            onClick()
        },
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
            .semantics {
                contentDescription = "Volunteer ${v.name}, rated ${v.rating}, specializing in ${v.specialties.joinToString(", ")}"
            }
    ) {
        Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
            Surface(color = tc.accent.copy(alpha = 0.2f), shape = CircleShape, modifier = Modifier.size(56.dp)) {
                Icon(Icons.Default.Person, null, tint = tc.accent, modifier = Modifier.padding(12.dp))
            }
            Spacer(Modifier.width(16.dp))
            Column(Modifier.weight(1f)) {
                Text(v.name, fontWeight = FontWeight.Bold, color = tc.text)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.Star, null, tint = Color(0xFFFFC107), modifier = Modifier.size(14.dp))
                    Text(" ${v.rating} • ${v.location} (${v.distanceKm}km)", color = tc.sub, fontSize = 12.sp)
                }
                Text(v.specialties.joinToString(", "), color = tc.accent, fontSize = 11.sp, maxLines = 1)
            }
            Icon(Icons.Default.KeyboardArrowRight, null, tint = tc.sub)
        }
    }
}

/**
 * A dialog showing detailed information about a volunteer.
 */
@Composable
fun VolunteerDetailDialog(v: Volunteer, tc: ThemeColors, onDismiss: () -> Unit) {
    val context = LocalContext.current
    val haptic = LocalHapticFeedback.current
    Dialog(onDismissRequest = onDismiss) {
        Surface(color = tc.card, shape = RoundedCornerShape(24.dp)) {
            Column(Modifier.padding(24.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.Person, null, modifier = Modifier.size(64.dp).background(tc.accent.copy(alpha = 0.1f), CircleShape).padding(12.dp), tint = tc.accent)
                    Spacer(Modifier.width(16.dp))
                    Column {
                        Text(v.name, fontWeight = FontWeight.ExtraBold, fontSize = 20.sp, color = tc.text)
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Star, null, tint = Color(0xFFFFC107), modifier = Modifier.size(16.dp))
                            Text(" ${v.rating} Rating", color = tc.sub, fontSize = 14.sp)
                        }
                    }
                }
                
                Text("Bio", fontWeight = FontWeight.Bold, color = tc.text)
                Text(v.bio, color = tc.sub, fontSize = 14.sp)
                
                Text("Helping in:", fontWeight = FontWeight.Bold, color = tc.text)
                FlowRow(mainAxisSpacing = 8.dp, crossAxisSpacing = 8.dp) {
                    v.specialties.forEach { spec ->
                        Surface(color = tc.card2, shape = CircleShape, border = BorderStroke(1.dp, tc.accent.copy(alpha = 0.3f))) {
                            Text(spec, Modifier.padding(horizontal = 12.dp, vertical = 6.dp), fontSize = 11.sp, color = tc.text)
                        }
                    }
                }
                
                Spacer(Modifier.height(8.dp))
                Button(onClick = {
                    vibrate(context)
                    /* Connect */
                }, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(12.dp)) {
                    Text("Request Assistance")
                }
                OutlinedButton(onClick = {
                    vibrate(context)
                    onDismiss()
                }, modifier = Modifier.fillMaxWidth(), border = BorderStroke(1.dp, tc.sub)) {
                    Text("Close", color = tc.text)
                }
            }
        }
    }
}

/**
 * A layout that flows its children across multiple rows if they exceed the width.
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun FlowRow(
    mainAxisSpacing: Dp = 0.dp,
    crossAxisSpacing: Dp = 0.dp,
    content: @Composable () -> Unit
) {
    FlowRow(
        horizontalArrangement = Arrangement.spacedBy(mainAxisSpacing),
        verticalArrangement = Arrangement.spacedBy(crossAxisSpacing),
        content = { content() }
    )
}

// ── Settings Screen ──────────────────────────────────────────────────────────

/**
 * Settings screen for app-wide configurations like theme and answer detail.
 */
@Composable
fun SettingsScreen(vm: InferenceViewModel, tc: ThemeColors) {
    val context = LocalContext.current
    val appTheme by vm.appTheme.collectAsState()
    val answerLength by vm.answerLength.collectAsState()
    var showUserGuide by remember { mutableStateOf(false) }
    val haptic = LocalHapticFeedback.current

    Column(Modifier.fillMaxSize().padding(20.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(24.dp)) {
        Text("Settings", fontSize = 28.sp, fontWeight = FontWeight.Bold, color = tc.text)

        // Appearance
        Column {
            Text("Appearance", fontWeight = FontWeight.Bold, color = tc.sub, fontSize = 12.sp)
            Spacer(Modifier.height(12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                AppTheme.entries.forEach { t ->
                    val sel = t == appTheme
                    Surface(
                        modifier = Modifier.weight(1f).clickable {
                            vibrate(context)
                            vm.setAppTheme(t)
                        },
                        color = if (sel) tc.accent.copy(alpha = 0.15f) else tc.card2,
                        shape = RoundedCornerShape(12.dp),
                        border = if (sel) BorderStroke(1.5.dp, tc.accent) else null
                    ) {
                        Column(Modifier.padding(12.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                            Text(t.emoji, fontSize = 20.sp)
                            Text(t.label, fontSize = 12.sp, color = tc.text, fontWeight = if (sel) FontWeight.Bold else FontWeight.Normal)
                        }
                    }
                }
            }
        }

        // Length
        Column {
            Text("Answer Detail", fontWeight = FontWeight.Bold, color = tc.sub, fontSize = 12.sp)
            Spacer(Modifier.height(12.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                AnswerLength.entries.forEach { a ->
                    val sel = a == answerLength
                    Surface(
                        modifier = Modifier.weight(1f).clickable {
                            vibrate(context)
                            vm.setAnswerLength(a)
                        },
                        color = if (sel) tc.accent.copy(alpha = 0.15f) else tc.card2,
                        shape = RoundedCornerShape(12.dp),
                        border = if (sel) BorderStroke(1.5.dp, tc.accent) else null
                    ) {
                        Column(Modifier.padding(12.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                            Text(a.emoji, fontSize = 20.sp)
                            Text(a.label, fontSize = 11.sp, color = tc.text, fontWeight = if (sel) FontWeight.Bold else FontWeight.Normal)
                        }
                    }
                }
            }
        }

        // Help & Support
        Column {
            Text("Support", fontWeight = FontWeight.Bold, color = tc.sub, fontSize = 12.sp)
            Spacer(Modifier.height(12.dp))
            Surface(
                modifier = Modifier.fillMaxWidth().clickable {
                    vibrate(context)
                    showUserGuide = true
                },
                color = tc.card2,
                shape = RoundedCornerShape(12.dp)
            ) {
                Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.Info, null, tint = tc.accent)
                    Spacer(Modifier.width(12.dp))
                    Text("User Guide", color = tc.text, fontWeight = FontWeight.Medium)
                    Spacer(Modifier.weight(1f))
                    Icon(Icons.Default.KeyboardArrowRight, null, tint = tc.sub)
                }
            }
        }

        Spacer(Modifier.height(12.dp))

        Text(
            text = "Offline AI • Powered by SmolVLM\nLocal Intelligence for Privacy",
            textAlign = TextAlign.Center,
            modifier = Modifier.fillMaxWidth(),
            fontSize = 12.sp,
            color = tc.accent.copy(alpha = 0.8f),
            lineHeight = 18.sp
        )

        Spacer(Modifier.height(8.dp))

        Text("App Version: 2.1.0-beta", textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth(), fontSize = 12.sp, color = tc.sub)
    }

    if (showUserGuide) {
        UserGuideDialog(tc) { showUserGuide = false }
    }
}

/**
 * Dialog displaying a guide on how to use the application.
 */
@Composable
fun UserGuideDialog(tc: ThemeColors, onDismiss: () -> Unit) {
    val context = LocalContext.current
    val haptic = LocalHapticFeedback.current
    Dialog(onDismissRequest = onDismiss) {
        Surface(color = tc.card, shape = RoundedCornerShape(24.dp), modifier = Modifier.fillMaxHeight(0.8f)) {
            Column(Modifier.padding(24.dp)) {
                Text("User Guide", fontWeight = FontWeight.ExtraBold, fontSize = 24.sp, color = tc.text)
                Spacer(Modifier.height(16.dp))
                Column(Modifier.weight(1f).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(16.dp)) {
                    GuideSectionDetail("🏠 Home Mode", "Upload images or videos from your gallery or capture them using the camera. Type or speak your question, and Mozix.ai will analyze the media to give you an answer.", tc)
                    GuideSectionDetail("🎥 Live Mode", "Real-time visual assistance. Mozix.ai 'watches' a few seconds of your camera feed to answer questions about what is happening right now.", tc)
                    GuideSectionDetail("🤝 Community", "Connect with local volunteers for shopping, medical help, or social activities. Register as a helper to give back to others.", tc)
                    GuideSectionDetail("⚙️ Settings", "Adjust the app theme (Light/Dark/System) and the detail level of AI responses (Concise/Standard/Detailed).", tc)
                    GuideSectionDetail("🔊 Voice Features", "Mozix.ai uses Text-to-Speech (TTS) to read answers aloud and Speech-to-Text for easy hands-free input.", tc)
                }
                Spacer(Modifier.height(16.dp))
                Button(onClick = {
                    vibrate(context)
                    onDismiss()
                }, modifier = Modifier.fillMaxWidth()) {
                    Text("Got it")
                }
            }
        }
    }
}

/**
 * Displays a section in the user guide with a title and description.
 */
@Composable
fun GuideSectionDetail(title: String, description: String, tc: ThemeColors) {
    Column {
        Text(title, fontWeight = FontWeight.Bold, color = tc.accent, fontSize = 16.sp)
        Text(description, color = tc.text, fontSize = 14.sp)
    }
}

// ── Shared UI Components ─────────────────────────────────────────────────────

/**
 * Dialog for choosing between taking a photo or recording a video.
 */
@Composable
fun CameraChoiceDialog(tc: ThemeColors, context: Context, photoLauncher: androidx.activity.result.ActivityResultLauncher<Uri>, videoLauncher: androidx.activity.result.ActivityResultLauncher<Uri>, onPhoto: (Uri) -> Unit, onVideo: (Uri) -> Unit, onDismiss: () -> Unit) {
    val haptic = LocalHapticFeedback.current
    Dialog(onDismissRequest = onDismiss) {
        Surface(color = tc.card, shape = RoundedCornerShape(20.dp)) {
            Column(Modifier.padding(24.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("Select Input", fontWeight = FontWeight.Bold, fontSize = 18.sp, color = tc.text)
                Button(onClick = {
                    vibrate(context)
                    if (androidx.core.content.ContextCompat.checkSelfPermission(context, android.Manifest.permission.CAMERA) == android.content.pm.PackageManager.PERMISSION_GRANTED) {
                        val uri = androidx.core.content.FileProvider.getUriForFile(context, "${context.packageName}.provider", java.io.File(context.cacheDir, "cam_${System.currentTimeMillis()}.jpg"))
                        onPhoto(uri); photoLauncher.launch(uri); onDismiss()
                    } else {
                        android.widget.Toast.makeText(context, "Camera permission required", android.widget.Toast.LENGTH_SHORT).show()
                    }
                }, modifier = Modifier.fillMaxWidth()) { Text("📷 Take Photo") }
                OutlinedButton(onClick = {
                    vibrate(context)
                    if (androidx.core.content.ContextCompat.checkSelfPermission(context, android.Manifest.permission.CAMERA) == android.content.pm.PackageManager.PERMISSION_GRANTED) {
                        val uri = androidx.core.content.FileProvider.getUriForFile(context, "${context.packageName}.provider", java.io.File(context.cacheDir, "vid_${System.currentTimeMillis()}.mp4"))
                        onVideo(uri); videoLauncher.launch(uri); onDismiss()
                    } else {
                        android.widget.Toast.makeText(context, "Camera permission required", android.widget.Toast.LENGTH_SHORT).show()
                    }
                }, modifier = Modifier.fillMaxWidth()) { Text("🎬 Record Video") }
            }
        }
    }
}

/**
 * Dialog for selecting the inference quality mode.
 */
@Composable
fun QualityDialog(current: QualityMode, tc: ThemeColors, onDismiss: () -> Unit, onSelect: (QualityMode) -> Unit) {
    val context = LocalContext.current
    val haptic = LocalHapticFeedback.current
    Dialog(onDismissRequest = onDismiss) {
        Surface(color = tc.card, shape = RoundedCornerShape(20.dp)) {
            Column(Modifier.padding(20.dp)) {
                Text("Quality Mode", color = tc.text, fontSize = 18.sp, fontWeight = FontWeight.Bold)
                Spacer(Modifier.height(16.dp))
                QualityMode.entries.forEach { mode ->
                    val sel = mode == current
                    Surface(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(vertical = 4.dp)
                            .clickable {
                                vibrate(context)
                                onSelect(mode); onDismiss()
                            },
                        color = if (sel) tc.accent.copy(alpha = 0.15f) else Color.Transparent,
                        shape = RoundedCornerShape(12.dp),
                        border = if (sel) BorderStroke(1.5.dp, tc.accent) else BorderStroke(1.dp, tc.sub.copy(alpha = 0.3f))
                    ) {
                        Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
                            Text(mode.emoji, fontSize = 26.sp)
                            Spacer(Modifier.width(12.dp))
                            Column(Modifier.weight(1f)) {
                                Text(mode.label, color = tc.text, fontWeight = FontWeight.SemiBold)
                                Text(mode.description, color = tc.sub, fontSize = 12.sp)
                            }
                            if (sel) Icon(Icons.Default.CheckCircle, null, tint = tc.accent)
                        }
                    }
                }
                Spacer(Modifier.height(12.dp)) // Extra safety margin at the bottom
            }
        }
    }
}

/**
 * A card representing an entry in the analysis history.
 */
@Composable
fun HistoryCard(entry: QAEntry, tc: ThemeColors, vm: InferenceViewModel, context: Context, onDelete: (QAEntry) -> Unit, onClick: (QAEntry) -> Unit) {
    val bm = vm.isBookmarked(entry)
    val haptic = LocalHapticFeedback.current
    Surface(
        color = tc.card2,
        shape = RoundedCornerShape(16.dp),
        modifier = Modifier
            .fillMaxWidth()
            .clickable {
                vibrate(context)
                onClick(entry)
            }
            .semantics {
                contentDescription = "Previous analysis: Question ${entry.question}, Answer ${entry.answer}"
            }
    ) {
        Row(Modifier.padding(12.dp), horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            entry.mediaUri?.let { uriStr ->
                Box(Modifier.size(80.dp).clip(RoundedCornerShape(12.dp)).background(tc.bg)) {
                    val bitmap = remember(uriStr) {
                        val uri = Uri.parse(uriStr)
                        try {
                            if (context.contentResolver.getType(uri)?.contains("video") == true) {
                                val r = MediaMetadataRetriever(); r.setDataSource(context, uri)
                                val b = r.getFrameAtTime(0); r.release(); b
                            } else {
                                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, uri)) { d, _, _ -> d.allocator = ImageDecoder.ALLOCATOR_SOFTWARE }
                                else {
                                    @Suppress("DEPRECATION")
                                    MediaStore.Images.Media.getBitmap(context.contentResolver, uri)
                                }
                            }
                        } catch (_: Exception) { null }
                    }
                    if (bitmap != null) Image(bitmap.asImageBitmap(), null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
                }
            }
                Column(Modifier.weight(1f)) {
                val qCount = entry.fullHistory?.size ?: 1
                Text(entry.question, color = tc.text, fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                Text(if (qCount > 1) "Conversation · $qCount questions" else entry.answer, color = tc.sub, fontSize = 13.sp, maxLines = 2, overflow = TextOverflow.Ellipsis)
                Spacer(Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    IconButton(onClick = {
                        vibrate(context)
                        vm.toggleBookmark(entry)
                    }, modifier = Modifier.size(24.dp)) {
                        Icon(Icons.Default.Star, null, tint = if (bm) Color(0xFFFFC107) else tc.sub.copy(alpha = 0.4f))
                    }
                    IconButton(onClick = {
                        vibrate(context)
                        onDelete(entry)
                    }, modifier = Modifier.size(24.dp)) {
                        Icon(Icons.Default.Delete, null, tint = Color.Red.copy(alpha = 0.6f))
                    }
                }
            }
        }
    }
}

/**
 * Extracts specific frames from a video URI as a list of Bitmaps.
 */
fun extractPreviewFrames(context: Context, uri: Uri, indices: List<Int>, total: Int): List<Bitmap> {
    val retriever = MediaMetadataRetriever(); val frames = mutableListOf<Bitmap>()
    try {
        retriever.setDataSource(context, uri); val d = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLong() ?: return emptyList()
        val interval = d * 1000L / total
        indices.forEach { idx -> 
            retriever.getFrameAtTime(idx * interval, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)?.let { full ->
                // Scale down for UI gallery to save memory (max 720p width)
                val targetWidth = 720
                val targetHeight = (full.height * (targetWidth.toFloat() / full.width)).toInt()
                val scaled = Bitmap.createScaledBitmap(full, targetWidth, targetHeight, true)
                if (scaled != full) full.recycle()
                frames.add(scaled)
            } 
        }
    } finally { retriever.release() }
    return frames
}

/**
 * Formats milliseconds into a mm:ss string.
 */
fun formatTime(ms: Int): String {
    val totalSeconds = ms / 1000
    val minutes = totalSeconds / 60
    val seconds = totalSeconds % 60
    return "%02d:%02d".format(minutes, seconds)
}
