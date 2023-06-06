package com.example.yolo_nas_compose

import android.graphics.RectF

data class Result(val classIndex: Int, val score: Float, val rectF: RectF)