package com.example.a1_cse535;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.os.Bundle;
import android.os.AsyncTask;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.MediaController;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.VideoView;
import android.content.Context;
//import java.io.File;
import java.io.*;
import java.net.*;
import java.net.URLConnection;
import javax.net.ssl.HttpsURLConnection;
//
//Instruction:
//0. After you choose gesture and Screen 2 is shown:
//1. Click Download button to download the .mp4
//2. Click Play button to play the gesture video
/*
Task on Screen 2:
    1. Here the video of an expert performing the gesture will be shown
    2. The video will have to be downloaded on the phone from the SignSavvy ASL gesture repository
https://www.signingsavvy.com/
    3. Screen 2 will have another button that should say “PRACTICE”.
On pressing this button, the user should be taken to Screen 3.
 */

public class Main2Activity extends AppCompatActivity {
    String selection;
    Boolean play;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);
        Bundle extras = getIntent().getExtras();
        selection = extras.getString("key");
        //Log.d("Msg tag",selection); //check for selection string transferred correctly

        // download video from URL according to user's choice
        Button downloadBt = (Button) findViewById(R.id.button6);
        downloadBt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                DownloadTask dt = new DownloadTask();
                dt.execute();
            }
        });

        // Button enables user go from Screen 2 to Screen 3
        Button practiceBt = (Button) findViewById(R.id.button2);
        practiceBt.setText("Practice");
        practiceBt.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Main2Activity.this, Main3Activity.class);
                intent.putExtra("key", selection);
                startActivity(intent);
            }
        });

        Button playBt = (Button) findViewById(R.id.button5);
        playBt.setOnClickListener(new View.OnClickListener(){
            public void onClick(View view){
                if (play == true){
                    VideoView vv = (VideoView) findViewById(R.id.videoView);
                    vv.setVideoPath(Environment.getExternalStorageDirectory()+"/my_folder/demo.mp4");
                    vv.start();
                    //video will display at screen 2 right here
                    Button playBt = (Button)findViewById(R.id.button5);// NOT USING RIGHT NOW
                    playBt.setEnabled(true);
                    //vvplay(play);
                }
            }
        });
// VV for playing local video
//        VideoView vv1 = (VideoView) findViewById(R.id.videoView);
//        String path = "android.resource://" + getPackageName() + "/" + R.raw.t;
//        Uri uri = Uri.parse(path);
//        vv1.setVideoURI(uri);
//        MediaController mc = (MediaController) new MediaController(this);
//        vv1.setMediaController(mc);
//        mc.setAnchorView(vv1);

    }

    public class DownloadTask extends AsyncTask<String, String, String> {

        @Override
        protected void onPreExecute() {
            Toast.makeText(getApplicationContext(), "onPreExecute() running", Toast.LENGTH_LONG).show();
        }

        protected String doInBackground(String... strings) {
            try {
                // init cannot be null or empty string with one space. Otherwise APP plays the wrong video
                URL url = new URL("https://www.signingsavvy.com/media/mp4-ld/6/6442.mp4");
                switch (selection.toLowerCase()){
                        case "buy":
                            // download buy video
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/6/6442.mp4");
                            break;
                        case "house":
                            // download house video
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/23/23234.mp4");
                            break;
                        case "fun":
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/22/22976.mp4");
                            break;
                        case "hope":
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/22/22197.mp4");
                            break;
                        case "arrive":
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/26/26971.mp4");
                            break;
                        case "really":
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/24/24977.mp4");
                            break;
                        case "read":
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/7/7042.mp4");
                            break;
                        case "lip":
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/26/26085.mp4");
                            break;
                        case "mouth":
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/22/22188.mp4");
                            break;
                        case "some":
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/23/23931.mp4");
                            break;
                        case "communicate":
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/22/22897.mp4");
                            break;
                        case "write":
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/27/27923.mp4");
                            break;
                        case "create":
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/22/22337.mp4");
                            break;
                        case "pretend":
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/25/25901.mp4");
                            break;
                        case "sister":
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/21/21587.mp4");
                            break;
                        case "man":
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/21/21568.mp4");
                            break;
                        case "one":
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/26/26492.mp4");
                            break;
                        case "drive":
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/23/23918.mp4");
                            break;
                        case "perfect":
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/24/24791.mp4");
                            break;
                        case "mother":
                            url = new URL("https://www.signingsavvy.com/media/mp4-ld/21/21571.mp4");
                            break;
                        default:
                            // consider a better solution for default case
                            // set the last one as the default case for now. deal with it later
                            //url = new URL("https://www.signingsavvy.com/media/mp4-ld/21/21571.mp4");
                            break;
                    }

                HttpsURLConnection urlConnection = (HttpsURLConnection) url.openConnection();
                urlConnection.setRequestMethod("POST");
                urlConnection.connect();
                File SDCardRoot = Environment.getExternalStorageDirectory(); // location where you want to store
                File directory = new File(SDCardRoot, "/my_folder/"); //create directory to keep your downloaded file

                if (!directory.exists()){
                    directory.mkdir();
                }

                File input_file = new File(directory, "demo.mp4");
                InputStream inputStream = new BufferedInputStream(url.openStream(),8192);
                byte[] data = new byte[1024];
                OutputStream outputStream = new FileOutputStream(input_file);
                byte[] buffer = new byte[1024];
                int bytesRead = 0;
                while ((bytesRead = inputStream.read(buffer, 0, buffer.length)) >= 0)
                {
                    outputStream.write(buffer, 0, bytesRead);
                }
                inputStream.close();
                outputStream.close();

            } catch(Exception e){
                e.printStackTrace();
            }

            return "Download Complete";
        }

        protected void onProgressUpdate(Integer... text) {

            //Toast.makeText(getApplicationContext(), "onProgressUpdate() running" , Toast.LENGTH_LONG).show();
        }

        @Override
        protected void onPostExecute(String text){
            Toast.makeText(getApplicationContext(), "Download Complete" , Toast.LENGTH_LONG).show();
            play = true;
        }
//            VideoView vv = (VideoView) findViewById(R.id.videoView); ↑
//            vv.setVideoPath(Environment.getExternalStorageDirectory()+"/my_folder/demo.mp4");
//            vv.start();
//            //video will display at screen 2 right here
//            Button playBt = (Button)findViewById(R.id.button5);// NOT USING RIGHT NOW
//            playBt.setEnabled(true);

//            VideoView vv1 = (VideoView) findViewById(R.id.videoView);
//            String path = "android.resource://" + getPackageName() + "/" + R.raw.t;
//            Uri uri = Uri.parse(path);
//            vv1.setVideoURI(uri);
//            MediaController mc = (MediaController) new MediaController(this);
//            vv1.setMediaController(mc);
//            mc.setAnchorView(vv1);

//        public void vvplay(Boolean str){
//            if (str == true){
//                playBt.setEnabled(true);
//            }
//        }
    }
}
