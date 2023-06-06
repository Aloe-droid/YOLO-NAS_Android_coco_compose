package com.example.yolo_nas_compose

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.util.Collections
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private val rectView = lazy { RectView(this) }
    private val dataProcess = lazy { DataProcess() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setPermissions()
        load()
        setContent {
            SetCamera()
        }
    }

    @Composable
    private fun SetCamera() {
        val processCameraProvider = ProcessCameraProvider.getInstance(this).get()
        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()
        val resolutionSelector = ResolutionSelector.Builder().setAspectRatioStrategy(
            AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY
        ).build()
        val preview =
            androidx.camera.core.Preview.Builder().setResolutionSelector(resolutionSelector).build()
        val analysis = ImageAnalysis.Builder().setResolutionSelector(resolutionSelector)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
        analysis.setAnalyzer(Executors.newSingleThreadExecutor()) {
            imageProcess(it)
            it.close()
        }
        val previewView = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
        }
        preview.setSurfaceProvider(previewView.surfaceProvider)
        processCameraProvider.bindToLifecycle(this, cameraSelector, preview, analysis)

        AndroidView(modifier = Modifier.fillMaxSize(), factory = { previewView })
        AndroidView(modifier = Modifier.fillMaxSize(), factory = { rectView.value })
    }


    // 이미지 처리
    private fun imageProcess(imageProxy: ImageProxy) {
        // YOLO_NAS_S : 0.41 ~ 0.46 초 소요, YOLO_NAS_S_QAT : 0.50 ~ 0.6 초 소요
        val bitmap = dataProcess.value.imgToBmp(imageProxy)
        val floatBuffer = dataProcess.value.bmpToFloatBuffer(bitmap)
        val inputName = ortSession.inputNames.iterator().next()
        // 모델의 입력 형태 [1 3 640 640] [배치 사이즈, 픽셀, 너비, 높이], 모델마다 다를 수 있음
        val shape = longArrayOf(
            DataProcess.BATCH_SIZE.toLong(),
            DataProcess.PIXEL_SIZE.toLong(),
            DataProcess.INPUT_SIZE.toLong(),
            DataProcess.INPUT_SIZE.toLong()
        )
        val inputTensor = OnnxTensor.createTensor(ortEnvironment, floatBuffer, shape)
        val resultTensor = ortSession.run(Collections.singletonMap(inputName, inputTensor))
        val output1 = (resultTensor.get(0).value as Array<*>)[0] as Array<*> // x, y, 너비, 높이
        val output2 = (resultTensor.get(1).value as Array<*>)[0] as Array<*> // 각 레이블 별 확률
        val results = dataProcess.value.outputToPredict(output1, output2)

        // 화면 표출
        rectView.value.transformRect(results)
        rectView.value.invalidate()
    }

    // 모델 불러오기
    private fun load() {
        dataProcess.value.loadModel(assets, filesDir.toString())
        dataProcess.value.loadLabel(assets)

        // 추론 객체
        ortEnvironment = OrtEnvironment.getEnvironment()
        // 모델 객체
        ortSession =
            ortEnvironment.createSession(
                filesDir.absolutePath.toString() + "/" + DataProcess.FILE_NAME
            )

        // 라벨링 배열 전달
        rectView.value.setClassLabel(dataProcess.value.classes)
    }

    // 권한 허용
    private fun setPermissions() {
        val requestPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestPermission()) {
                if (!it) {
                    Toast.makeText(this, "권한을 허용 하지 않으면 사용할 수 없습니다!", Toast.LENGTH_SHORT).show()
                    finish()
                }
            }

        val permissions = listOf(Manifest.permission.CAMERA)

        permissions.forEach {
            if (ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED) {
                requestPermissionLauncher.launch(it)
            }
        }
    }
}