package com.example.a1_cse535;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Environment;
import android.provider.MediaStore;
//import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.PrecomputedText;
import android.util.Log;
import android.view.View;
import android.webkit.CookieManager;
import android.webkit.ValueCallback;
import android.webkit.WebView;
import android.widget.Button;
import android.widget.Toast;
import android.widget.VideoView;
import java.io.File;
import java.net.URL;
import java.net.URLConnection;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.net.HttpURLConnection;

/*
In this screen the camera interface should be opened for the user to record the practice gesture.
The video should be captured for 5 s. The video should be saved with the following format for the video filename
GESTURE_PRACTICE_(Practice Number)_USERLASTNAME.mp4
There should be another button in the Screen 3 with the name UPLOAD
On pressing this button, the user should be able to upload the gesture to a local server. Moreover, clicking this button will also take back the user to screen 1.
Local Server: Develop your own local server in your own laptop.
Usage of the mobile application:

 */

//import android.widget.MediaController;
//import android.net.Uri;
public class Main3Activity extends AppCompatActivity {
    String selection;
    private static final int VIDEO_CAPTURE = 101;
    private Uri fileUri;
    private static int count = 0;
    private static String tempGes = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main3);
        Bundle extras = getIntent().getExtras();
        selection = extras.getString("key");

        Button camBt = (Button) findViewById(R.id.button4);
        //a button that opens camera for 5s

        if (!tempGes.equals(selection)){
            count = 0;
            tempGes = selection;
        }

        if (!hasCamera()) {
            camBt.setEnabled(false);
        }

        camBt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startRecording();
            }
        });

        //video saved as GESTURE_PRACTICE_(Practice Number)_USERLASTNAME.mp4
        // button upload
        Button uploadBt = (Button) findViewById(R.id.button3);
        uploadBt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                UploadTask uploadTask = new UploadTask();
                uploadTask.execute();
                Intent intent = new Intent(Main3Activity.this, MainActivity.class);
                startActivity(intent);
            }
        });
    }

    private boolean hasCamera() {
        if (getPackageManager().hasSystemFeature(
                PackageManager.FEATURE_CAMERA_ANY)) {
            return true;
        } else {
            return false;
        }
    }

    //video saved as GESTURE_PRACTICE_(Practice Number)_USERLASTNAME.mp4
    public void startRecording(){
        count++;
        File mediaFile = new
                File(Environment.getExternalStorageDirectory().getAbsolutePath()
                + "/my_folder/" + tempGes.toUpperCase() + "_PRACTICE_(" + count + ")_WU.mp4");

        Intent intent = new Intent(MediaStore.ACTION_VIDEO_CAPTURE);
        intent.putExtra(MediaStore.EXTRA_DURATION_LIMIT,5);
        fileUri = Uri.fromFile(mediaFile);

        intent.putExtra(MediaStore.EXTRA_OUTPUT, fileUri);
        startActivityForResult(intent, VIDEO_CAPTURE);
    }



    public class UploadTask extends AsyncTask<String, String, String>{

        @Override
        protected String doInBackground(String... strings) {
            try {

                String url = "http://10.218.107.121/cse535/upload_video.php";
                String charset = "UTF-8";
                String group_id = "12";
                String ASUid = "1209740191";
                String accept = "1";

                //String practiceNum = "1";
                int practiceNum = 1; // edit later bc 1 2 3 should all exist
                selection.toUpperCase();
                //GESTURE_PRACTICE_(Practice Number)_USERLASTNAME.mp4
                //File videoFile = new File(Environment.getExternalStorageDirectory()+"/my_folder/" + selection + "_PRACTICE_(" + practiceNum + ")_WU.mp4");

                //+ "/my_folder/" + tempGes.toUpperCase() + "_PRACTICE_(" + count + ")_WU.mp4");
                File videoFile = new File(Environment.getExternalStorageDirectory()+"/my_folder/" + tempGes.toUpperCase() + "_PRACTICE_(" + count + ")_WU.mp4");
                String boundary = Long.toHexString(System.currentTimeMillis()); // Just generate some unique random value.
                String CRLF = "\r\n"; // Line separator required by multipart/form-data.

                URLConnection connection;

                connection = new URL(url).openConnection();
                connection.setDoOutput(true);
                connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);

                try (
                        OutputStream output = connection.getOutputStream();
                        PrintWriter writer = new PrintWriter(new OutputStreamWriter(output, charset), true);
                ) {
                    // Send normal accept.
                    writer.append("--" + boundary).append(CRLF);
                    writer.append("Content-Disposition: form-data; name=\"accept\"").append(CRLF);
                    writer.append("Content-Type: text/plain; charset=" + charset).append(CRLF);
                    writer.append(CRLF).append(accept).append(CRLF).flush();

                    // Send normal accept.
                    writer.append("--" + boundary).append(CRLF);
                    writer.append("Content-Disposition: form-data; name=\"id\"").append(CRLF);
                    writer.append("Content-Type: text/plain; charset=" + charset).append(CRLF);
                    writer.append(CRLF).append(ASUid).append(CRLF).flush();

                    // Send normal accept.
                    writer.append("--" + boundary).append(CRLF);
                    writer.append("Content-Disposition: form-data; name=\"group_id\"").append(CRLF);
                    writer.append("Content-Type: text/plain; charset=" + charset).append(CRLF);
                    writer.append(CRLF).append(group_id).append(CRLF).flush();


                    // Send video file.
                    writer.append("--" + boundary).append(CRLF);
                    writer.append("Content-Disposition: form-data; name=\"uploaded_file\"; filename=\"" + videoFile.getName() + "\"").append(CRLF);
                    writer.append("Content-Type: video/mp4; charset=" + charset).append(CRLF); // Text file itself must be saved in this charset!
                    writer.append(CRLF).flush();


                    FileInputStream vf = new FileInputStream(videoFile);
                    try {
                        byte[] buffer = new byte[1024];
                        int bytesRead = 0;
                        while ((bytesRead = vf.read(buffer, 0, buffer.length)) >= 0)
                        {
                            output.write(buffer, 0, bytesRead);

                        }
                        //   output.close();
                        //Toast.makeText(getApplicationContext(),"Read Done", Toast.LENGTH_LONG).show();
                    }catch (Exception exception)
                    {


                        //Toast.makeText(getApplicationContext(),"output exception in catch....."+ exception + "", Toast.LENGTH_LONG).show();
                        Log.d("Error", String.valueOf(exception));
                        publishProgress(String.valueOf(exception));
                        // output.close();

                    }

                    output.flush(); // Important before continuing with writer!
                    writer.append(CRLF).flush(); // CRLF is important! It indicates end of boundary.


                    // End of multipart/form-data.
                    writer.append("--" + boundary + "--").append(CRLF).flush();
                } catch (UnsupportedEncodingException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                // Request is lazily fired whenever you need to obtain information about response.
                int responseCode = ((HttpURLConnection) connection).getResponseCode();
                System.out.println(responseCode); // Should be 200

            } catch (IOException e) {
                e.printStackTrace();
            }

            return null;
        }


        @Override
        protected void onProgressUpdate(String... text) {
            Toast.makeText(getApplicationContext(), "In Background Task " + text[0], Toast.LENGTH_LONG).show();
        }

    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == VIDEO_CAPTURE) {
            if (resultCode == RESULT_OK) {
                Toast.makeText(this, "Video has been saved to:\n" +
                        data.getData(), Toast.LENGTH_LONG).show();
            } else if (resultCode == RESULT_CANCELED) {
                Toast.makeText(this, "Video recording cancelled.",
                        Toast.LENGTH_LONG).show();
            } else {
                Toast.makeText(this, "Failed to record video",
                        Toast.LENGTH_LONG).show();
            }
        }
    }

}
