package com.example.fakenewsapp;

import android.content.Context;
import android.content.res.AssetFileDescriptor;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class FakeNewsDetector {
    private Interpreter interpreter;

    // Constructor to load the TFLite model
    public FakeNewsDetector(Context context) {
        try {
            interpreter = new Interpreter(loadModelFile(context, "model.tflite"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Method to load the TFLite model from assets
    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // Method to run inference
    public float predict(ByteBuffer input) {
        float[][] output = new float[1][1]; // Assuming model outputs a single float value
        interpreter.run(input, output);
        return output[0][0]; // Return the prediction result
    }
}