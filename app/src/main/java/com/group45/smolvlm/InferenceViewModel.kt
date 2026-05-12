package com.group45.smolvlm
import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import android.app.Application
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.provider.MediaStore
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

// ── Quality Mode ───────────────────────────────────────────────────────────────
enum class QualityMode(
    val label:       String,
    val numFrames:   Int,
    val topK:        Int,
    val emoji:       String,
    val description: String,
    val maxTokens:   Int
) {
    FAST    ("Fast",     numFrames = 12, topK = 4,  emoji = "⚡", description = "12 Frames (Top 4) · Fast",   maxTokens = 80),
    BALANCED("Balanced", numFrames = 16, topK = 6,  emoji = "⚖️", description = "16 Frames (Top 6) · Good",   maxTokens = 150),
    ACCURATE("Accurate", numFrames = 24, topK = 8,  emoji = "🎯", description = "24 Frames (Top 8) · High",   maxTokens = 250),
    ULTRA   ("Ultra",    numFrames = 32, topK = 12, emoji = "🔬", description = "32 Frames (Top 12) · Max",   maxTokens = 400)
}

// ── App Theme ──────────────────────────────────────────────────────────────────
enum class AppTheme(val label: String, val emoji: String) {
    DARK ("Dark",  "🌙"),
    LIGHT("Light", "☀️"),
    SYSTEM("Auto", "📱")
}

// ── Answer Length Preference ──────────────────────────────────────────────────
enum class AnswerLength(val label: String, val emoji: String, val maxTokens: Int, val instruction: String) {
    SHORT ("Short",    "✂️",  60,  "Answer in one short sentence."),
    MEDIUM("Medium",   "📝",  120, "Provide a concise answer."),
    LONG  ("Detailed", "📖",  300, "Provide a detailed description.")
}

class InferenceViewModel(app: Application) : AndroidViewModel(app) {

    private val engine = SmolVLMEngine(app)
    private val prefs = app.getSharedPreferences("smolvlm_prefs", Context.MODE_PRIVATE)
    private val gson = Gson()

    // ── Home Screen States ─────────────────────────────────────────────────────
    private val _homeState = MutableStateFlow<UiState>(UiState.Idle)
    val homeState: StateFlow<UiState> = _homeState

    private val _homeProgress = MutableStateFlow(0f)
    val homeProgress: StateFlow<Float> = _homeProgress

    private val _homePartialResult = MutableStateFlow("")
    val homePartialResult: StateFlow<String> = _homePartialResult

    private val _homeLastStats = MutableStateFlow<InferenceStats?>(null)
    val homeLastStats: StateFlow<InferenceStats?> = _homeLastStats

    private val _homeQuestion = MutableStateFlow("")
    val homeQuestion: StateFlow<String> = _homeQuestion

    private val _selectedVideo = MutableStateFlow<Uri?>(null)
    val selectedVideo: StateFlow<Uri?> = _selectedVideo

    private val _selectedImage = MutableStateFlow<Uri?>(null)
    val selectedImage: StateFlow<Uri?> = _selectedImage

    // ── Live Screen States ─────────────────────────────────────────────────────
    private val _liveState = MutableStateFlow<UiState>(UiState.Idle)
    val liveState: StateFlow<UiState> = _liveState

    private val _liveProgress = MutableStateFlow(0f)
    val liveProgress: StateFlow<Float> = _liveProgress

    private val _livePartialResult = MutableStateFlow("")
    val livePartialResult: StateFlow<String> = _livePartialResult

    private val _liveLastStats = MutableStateFlow<InferenceStats?>(null)
    val liveLastStats: StateFlow<InferenceStats?> = _liveLastStats

    // ── Settings & Global States ───────────────────────────────────────────────
    private val _qualityMode = MutableStateFlow(QualityMode.BALANCED)
    val qualityMode: StateFlow<QualityMode> = _qualityMode

    private val _appTheme = MutableStateFlow(AppTheme.DARK)
    val appTheme: StateFlow<AppTheme> = _appTheme

    private val _answerLength = MutableStateFlow(AnswerLength.MEDIUM)
    val answerLength: StateFlow<AnswerLength> = _answerLength

    private val _history = MutableStateFlow<List<QAEntry>>(loadHistory())
    val history: StateFlow<List<QAEntry>> = _history

    private val _bookmarks = MutableStateFlow<List<QAEntry>>(loadBookmarks())
    val bookmarks: StateFlow<List<QAEntry>> = _bookmarks

    private val _currentChatHistory = MutableStateFlow<List<Pair<String, String>>>(emptyList())
    val currentChatHistory: StateFlow<List<Pair<String, String>>> = _currentChatHistory

    private val _isNarrating = MutableStateFlow(false)
    val isNarrating: StateFlow<Boolean> = _isNarrating

    private val _isNarratorScanning = MutableStateFlow(false)
    val isNarratorScanning: StateFlow<Boolean> = _isNarratorScanning

    private val _narratorText = MutableStateFlow("")
    val narratorText: StateFlow<String> = _narratorText

    private var homeJob: Job? = null
    private var liveJob: Job? = null
    private var liveCaptureJob: Job? = null
    private var narrationJob: Job? = null

    private fun saveHistory(list: List<QAEntry>) {
        prefs.edit().putString("history", gson.toJson(list)).apply()
    }

    private fun loadHistory(): List<QAEntry> {
        val json = prefs.getString("history", null) ?: return emptyList()
        return try {
            val type = object : TypeToken<List<QAEntry>>() {}.type
            gson.fromJson(json, type)
        } catch (_: Exception) { emptyList() }
    }

    private fun saveBookmarks(list: List<QAEntry>) {
        prefs.edit().putString("bookmarks", gson.toJson(list)).apply()
    }

    private fun loadBookmarks(): List<QAEntry> {
        val json = prefs.getString("bookmarks", null) ?: return emptyList()
        return try {
            val type = object : TypeToken<List<QAEntry>>() {}.type
            gson.fromJson(json, type)
        } catch (_: Exception) { emptyList() }
    }

    // ── INPUT HANDLING ─────────────────────────────────────────────────────────

    fun setVideo(uri: Uri) {
        homeJob?.cancel()
        _selectedVideo.value = uri
        _selectedImage.value = null
        _homeState.value = UiState.Idle
        _homeProgress.value = 0f
        _homeLastStats.value = null
        _homeQuestion.value = ""
        _currentChatHistory.value = emptyList()
    }

    fun setImage(uri: Uri) {
        homeJob?.cancel()
        _selectedImage.value = uri
        _selectedVideo.value = null
        _homeState.value = UiState.Idle
        _homeProgress.value = 0f
        _homeLastStats.value = null
        _homeQuestion.value = ""
        _currentChatHistory.value = emptyList()
    }

    fun setHomeQuestion(q: String) { _homeQuestion.value = q }

    fun clearChatHistory() { _currentChatHistory.value = emptyList() }

    fun clearNarratorText() { _narratorText.value = "" }

    fun setQualityMode(mode: QualityMode) { _qualityMode.value = mode }
    fun setAppTheme(theme: AppTheme)       { _appTheme.value    = theme }
    fun setAnswerLength(len: AnswerLength)  { _answerLength.value = len }

    fun setLiveCaptureJob(job: Job) {
        liveCaptureJob?.cancel()
        liveCaptureJob = job
    }

    fun setLiveStatus(msg: String, progress: Float = 0f) {
        _liveState.value = UiState.Loading(msg)
        _liveProgress.value = progress
    }

    fun cancelHomeInference() {
        engine.requestStop()
        homeJob?.cancel()
        _homeState.value = UiState.Idle
        _homeProgress.value = 0f
    }

    fun cancelLiveInference() {
        engine.requestStop()
        liveCaptureJob?.cancel()
        liveJob?.cancel()
        _liveState.value = UiState.Idle
        _liveProgress.value = 0f
    }

    // ── BOOKMARK ───────────────────────────────────────────────────────────────

    fun toggleBookmark(entry: QAEntry) {
        val current = _bookmarks.value
        val newList = if (current.any { (it.question == entry.question) && (it.answer == entry.answer) }) {
            current.filterNot { (it.question == entry.question) && (it.answer == entry.answer) }
        } else {
            current + entry
        }
        _bookmarks.value = newList
        saveBookmarks(newList)
    }

    fun isBookmarked(entry: QAEntry): Boolean =
        _bookmarks.value.any { it.question == entry.question && it.answer == entry.answer }

    // ── MAIN ENTRY (HOME) ──────────────────────────────────────────────────────

    fun runInference(question: String) {
        if (question.isBlank()) return
        if (_homeState.value is UiState.Loading) return
        _homePartialResult.value = ""

        val video = _selectedVideo.value
        val image = _selectedImage.value
        val mode  = _qualityMode.value
        val ans   = _answerLength.value
        val history = _currentChatHistory.value

        if (video != null) runOnVideo(video, question, mode, ans, history)
        else if (image != null) runOnImage(image, question, ans, history)
    }

    private fun runOnVideo(uri: Uri, question: String, mode: QualityMode, ans: AnswerLength, history: List<Pair<String, String>>) {
        // FIX: cap effectiveTokens so Ultra/Accurate modes don't pass 300-400 tokens
        // and blow past the sentence-level early-stop guard in the engine.
        // maxTokens in QualityMode are aspirational upper bounds; actual output length
        // is controlled by MAX_SENTENCES (3 sentences). Cap at 120 to keep responses focused.
        val effectiveTokens = minOf(maxOf(mode.maxTokens, ans.maxTokens), 120)
        _homeState.value = UiState.Loading("Extracting ${mode.numFrames} frames…")
        _homeProgress.value = 0.1f

        homeJob?.cancel()
        homeJob = viewModelScope.launch(Dispatchers.IO) {
            val startMs = System.currentTimeMillis()
            try {
                engine.setMode(mode.numFrames, mode.topK, effectiveTokens)
                val result = engine.runOnVideo(
                    videoUri = uri,
                    question = question,
                    history = history,
                    instruction = ans.instruction,
                    temperature = 0.1f, // Precise video analysis
                    onStatus = { msg -> _homeState.value = UiState.Loading(msg) },
                    onToken = { token -> _homePartialResult.value += token }
                )

                if (!isActive) return@launch
                _homeProgress.value = 1f

                withContext(Dispatchers.Main) {
                    when (result) {
                        is InferenceResult.Success -> {
                            val currentChat = _currentChatHistory.value + (question to result.answer)
                            _currentChatHistory.value = currentChat
                            val entry = QAEntry(
                                question      = question,
                                answer        = result.answer,
                                elapsedMs     = result.elapsedMs,
                                mode          = mode.label,
                                selectedFrames = result.selectedFrames,
                                totalFrames   = result.totalFrames,
                                mediaUri      = uri.toString(),
                                fullHistory   = currentChat
                            )
                            val newHistory = _history.value + entry
                            _history.value = newHistory
                            saveHistory(newHistory)

                            _homeLastStats.value = InferenceStats(
                                framesExtracted  = mode.numFrames,
                                framesSelected   = result.selectedFrames.size,
                                totalFrames      = result.totalFrames,
                                inferenceMs      = result.elapsedMs,
                                totalPipelineMs  = System.currentTimeMillis() - startMs,
                                mode             = mode.label,
                                tokensGenerated  = result.answer.split(" ").size
                            )
                            _homeState.value = UiState.Done(result)
                        }
                        is InferenceResult.Error -> _homeState.value = UiState.Failure(result.message)
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    if (e is CancellationException) {
                        _homeState.value = UiState.Idle
                    } else {
                        _homeState.value = UiState.Failure(e.message ?: "Analysis failed")
                    }
                    _homeProgress.value = 0f
                }
            }
        }
    }

    private fun runOnImage(uri: Uri, question: String, ans: AnswerLength, history: List<Pair<String, String>>) {
        _homeState.value = UiState.Loading("Processing image…")
        _homeProgress.value = 0.2f

        homeJob?.cancel()
        homeJob = viewModelScope.launch(Dispatchers.IO) {
            val startMs = System.currentTimeMillis()
            try {
                val bmp: Bitmap = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    val source = ImageDecoder.createSource(getApplication<Application>().contentResolver, uri)
                    ImageDecoder.decodeBitmap(source) { decoder, _, _ -> decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE }
                } else {
                    @Suppress("DEPRECATION")
                    MediaStore.Images.Media.getBitmap(getApplication<Application>().contentResolver, uri)
                }

                engine.setMode(1, 1, ans.maxTokens)
                val result = engine.runOnImage(
                    bitmap = bmp,
                    question = question,
                    history = history,
                    instruction = ans.instruction,
                    temperature = 0.1f, // Precise image analysis
                    onStatus = { msg -> _homeState.value = UiState.Loading(msg) },
                    onToken = { token -> _homePartialResult.value += token }
                )

                if (!isActive) return@launch
                _homeProgress.value = 1f

                withContext(Dispatchers.Main) {
                    when (result) {
                        is InferenceResult.Success -> {
                            val currentChat = _currentChatHistory.value + (question to result.answer)
                            _currentChatHistory.value = currentChat
                            val entry = QAEntry(
                                question       = question,
                                answer         = result.answer,
                                elapsedMs      = result.elapsedMs,
                                mode           = "Image",
                                mediaUri       = uri.toString(),
                                fullHistory    = currentChat
                            )
                            val newHistory = _history.value + entry
                            _history.value = newHistory
                            saveHistory(newHistory)

                            _homeLastStats.value = InferenceStats(
                                framesExtracted  = 1,
                                framesSelected   = 1,
                                totalFrames      = 1,
                                inferenceMs      = result.elapsedMs,
                                totalPipelineMs  = System.currentTimeMillis() - startMs,
                                mode             = "Image",
                                tokensGenerated  = result.answer.split(" ").size
                            )
                            _homeState.value = UiState.Done(result)
                        }
                        is InferenceResult.Error -> _homeState.value = UiState.Failure(result.message)
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    if (e is CancellationException) _homeState.value = UiState.Idle
                    else _homeState.value = UiState.Failure(e.message ?: "Error decoding image")
                    _homeProgress.value = 0f
                }
            }
        }
    }

    // ── LIVE VIDEO INFERENCE ──────────────────────────────────────────────────

    fun runOnLiveVideo(frames: List<Bitmap>, question: String, history: List<Pair<String, String>> = emptyList()) {
        val ans = _answerLength.value
        val liveMaxTokens = maxOf(130, ans.maxTokens)
        _liveState.value = UiState.Loading("Analysing Live Feed…")
        _liveProgress.value = 0.5f
        _livePartialResult.value = ""

        liveJob?.cancel()
        liveJob = viewModelScope.launch(Dispatchers.IO) {
            val startMs = System.currentTimeMillis()
            try {
                // topK = 2 is enough for quick live identification
                engine.setMode(frames.size, topK = 2, maxTokens = liveMaxTokens)
                val result = engine.runOnFrameList(
                    frames = frames,
                    question = question,
                    history = history,
                    instruction = "Answer as concisely as possible. ${ans.instruction}",
                    startTimeMs = startMs,
                    temperature = 0.2f, // Lower temperature for more stable live output
                    onStatus = { msg -> _liveState.value = UiState.Loading(msg) },
                    onToken = { token -> _livePartialResult.value += token }
                )

                if (!isActive) return@launch
                _liveProgress.value = 1f

                withContext(Dispatchers.Main) {
                    when (result) {
                        is InferenceResult.Success -> {
                            val currentChat = _currentChatHistory.value + (question to result.answer)
                            _currentChatHistory.value = currentChat
                            val entry = QAEntry(
                                question      = question,
                                answer        = result.answer,
                                elapsedMs     = result.elapsedMs,
                                mode          = "Live",
                                selectedFrames = result.selectedFrames,
                                totalFrames   = result.totalFrames,
                                mediaUri      = null,
                                fullHistory   = currentChat
                            )
                            val newHistory = _history.value + entry
                            _history.value = newHistory
                            saveHistory(newHistory)

                            _liveLastStats.value = InferenceStats(
                                framesExtracted  = frames.size,
                                framesSelected   = result.selectedFrames.size,
                                totalFrames      = result.totalFrames,
                                inferenceMs      = result.elapsedMs,
                                totalPipelineMs  = System.currentTimeMillis() - startMs,
                                mode             = "Live",
                                tokensGenerated  = result.answer.split(" ").size
                            )
                            _liveState.value = UiState.Done(result)
                        }
                        is InferenceResult.Error -> _liveState.value = UiState.Failure(result.message)
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    if (e is CancellationException) _liveState.value = UiState.Idle
                    else _liveState.value = UiState.Failure(e.message ?: "Live inference failed")
                }
            }
        }
    }

    // ── PROACTIVE NARRATOR ────────────────────────────────────────────────────

    fun toggleNarration(getFrames: suspend () -> List<Bitmap>, onSpeak: (String) -> Unit) {
        if (_isNarrating.value) {
            _isNarrating.value = false
            _narratorText.value = ""
            narrationJob?.cancel()
        } else {
            _isNarrating.value = true
            val narrationQueries = listOf(
                "Describe the environment and the most prominent objects in this scene.",
                "Summarize what is happening in front of the camera right now.",
                "Identify any key objects, people or obstacles in the view.",
                "What are the most important details of this scene?",
                "Provide a quick overview of the current surroundings."
            )
            var queryIndex = 0

            narrationJob = viewModelScope.launch(Dispatchers.IO) {
                while (isActive) {
                    try {
                        _isNarratorScanning.value = true
                        val frames = getFrames()
                        _isNarratorScanning.value = false

                        if (frames.isNotEmpty()) {
                            val currentQuery = narrationQueries[queryIndex % narrationQueries.size]
                            queryIndex++

                            engine.setMode(frames.size, topK = 2, maxTokens = 80)
                            val result = engine.runOnFrameList(
                                frames = frames,
                                question = currentQuery,
                                history = emptyList(),
                                instruction = "Describe strictly and concisely in one sentence.",
                                isNarrator = true,
                                temperature = 0.3f // More factual narrator
                            )
                            if (result is InferenceResult.Success) {
                                withContext(Dispatchers.Main) {
                                    _narratorText.value = result.answer
                                    onSpeak(result.answer)
                                }
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("Narration", "Failed step", e)
                    }
                    delay(8500)
                }
            }
        }
    }

    // ── RESET ──────────────────────────────────────────────────────────────────

    fun resetHome() {
        homeJob?.cancel()
        _homeState.value = UiState.Idle
        _selectedVideo.value = null
        _selectedImage.value = null
        _homeProgress.value = 0f
        _homeLastStats.value = null
        _currentChatHistory.value = emptyList()
    }

    fun resetLive() {
        engine.requestStop()
        liveCaptureJob?.cancel()
        liveJob?.cancel()
        narrationJob?.cancel()
        _isNarrating.value = false
        _isNarratorScanning.value = false
        _narratorText.value = ""
        liveCaptureJob = null
        liveJob = null
        _liveState.value = UiState.Idle
        _liveProgress.value = 0f
        _liveLastStats.value = null
        _livePartialResult.value = ""
        _currentChatHistory.value = emptyList()
    }

    fun resetLiveStateOnly() {
        _liveState.value = UiState.Idle
        _liveProgress.value = 0f
    }

    fun clearHistory() {
        _history.value = emptyList()
        saveHistory(emptyList())
    }

    fun clearBookmarks() {
        _bookmarks.value = emptyList()
        saveBookmarks(emptyList())
    }

    fun deleteHistoryEntry(entry: QAEntry) {
        val newList = _history.value - entry
        _history.value = newList
        saveHistory(newList)
    }

    fun restoreEntry(entry: QAEntry) {
        homeJob?.cancel()
        val uri = entry.mediaUri?.let { Uri.parse(it) }
        if (entry.mode == "Image") {
            _selectedImage.value = uri
            _selectedVideo.value = null
        } else {
            _selectedVideo.value = uri
            _selectedImage.value = null
        }

        _homeState.value = UiState.Done(
            InferenceResult.Success(
                answer = entry.answer,
                selectedFrames = entry.selectedFrames,
                totalFrames = entry.totalFrames,
                elapsedMs = entry.elapsedMs
            )
        )

        _homeLastStats.value = InferenceStats(
            framesExtracted = entry.totalFrames,
            framesSelected = entry.selectedFrames.size,
            totalFrames = entry.totalFrames,
            inferenceMs = entry.elapsedMs,
            totalPipelineMs = entry.elapsedMs,
            mode = entry.mode,
            tokensGenerated = entry.answer.split(" ").size
        )
        // FIX: do NOT restore fullHistory into _currentChatHistory.
        // Previously restoring an old entry loaded its entire Q&A history into the active chat
        // state. The next runInference() call then passed that stale history (including completely
        // unrelated questions like "count the word the") into the engine as the `history` param,
        // causing it to answer the old question instead of the new video. Restored entries are
        // display-only — chat context must always start fresh for the next query.
        _currentChatHistory.value = emptyList()
    }

    override fun onCleared() {
        homeJob?.cancel()
        liveJob?.cancel()
        liveCaptureJob?.cancel()
        engine.close()
        super.onCleared()
    }
}

// ── DATA CLASSES ───────────────────────────────────────────────────────────────

data class QAEntry(
    val question:       String,
    val answer:         String,
    val elapsedMs:      Long,
    val mode:           String,
    val selectedFrames: List<Int> = emptyList(),
    val totalFrames:    Int = 0,
    val mediaUri:       String? = null,
    val fullHistory:    List<Pair<String, String>>? = null
)

data class InferenceStats(
    val framesExtracted:  Int,
    val framesSelected:   Int,
    val totalFrames:      Int,
    val inferenceMs:      Long,
    val totalPipelineMs:  Long,
    val mode:             String,
    val tokensGenerated:  Int
)

// ── UI STATE ───────────────────────────────────────────────────────────────────

sealed class UiState {
    object Idle : UiState()
    data class Loading(val message: String) : UiState()
    data class Done(val result: InferenceResult.Success) : UiState()
    data class Failure(val message: String) : UiState()
}
