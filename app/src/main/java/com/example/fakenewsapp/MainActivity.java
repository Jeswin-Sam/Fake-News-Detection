package com.example.fakenewsapp;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import com.example.fakenewsapp.ml.FakeNewsDetectionModelV1;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.Charset;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        EditText input = findViewById(R.id.Input_box);
        Button button = findViewById(R.id.check_button);
        TextView output = findViewById(R.id.output_text);

        FakeNewsDetector fakeNewsDetector = new FakeNewsDetector(this);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                String input_text = input.getText().toString();
                Charset charset = Charset.forName("UTF-8");

                ByteBuffer byteBuffer = ByteBuffer.wrap(input_text.getBytes(charset));

                try {
                    FakeNewsDetectionModelV1 model = FakeNewsDetectionModelV1.newInstance(MainActivity.this);

                    // Creates inputs for reference.
                    TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 250}, DataType.FLOAT32);
                    inputFeature0.loadBuffer(byteBuffer);

                    // Runs model inference and gets result.
                    FakeNewsDetectionModelV1.Outputs outputs = model.process(inputFeature0);
                    TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

                    // Releases model resources if no longer used.
                    model.close();
                } catch (IOException e) {
                    // TODO Handle the exception
                }



//                FakeNewsModel model = new FakeNewsModel(this, "fake_news_detection_model_v1.tflite");
//
//                // Example input. Replace this with actual preprocessed input.
//                float[] input = new float[]{/* your preprocessed input here */};
//
//                float[] prediction = model.predict(input);
//
//                // Handle your prediction result here
//                output.setText(Float.toString(prediction[0]));
//
        };
    });

}
}