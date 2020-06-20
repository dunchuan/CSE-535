package com.example.a1_cse535;

//import android.support.v7.app.AppCompatActivity;

import androidx.appcompat.app.AppCompatActivity;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {
private Spinner spinner;
private static final String [] ges_list = new String[] {"Choose a gesture", "buy", "house", "fun", "hope", "arrive", "really",
            "read", "lip", "mouth", "some", "communicate", "write", "create", "pretend", "sister",
            "man", "one", "drive", "perfect", "mother"};
//private List<String> ges_list;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //temp button
//        Button bt_no_use = (Button) findViewById(R.id.button);
//        bt_no_use.setText("Jump to Screen 2");
//        bt_no_use.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//                Intent intent1 = new Intent(MainActivity.this, Main2Activity.class);
//                startActivity(intent1);
//            }
//        });

        // Setup spinner for dropdown menu
        spinner  = (Spinner) findViewById(R.id.spinner);
        final ArrayAdapter<String> adapter = new ArrayAdapter<String> (this, android.R.layout.simple_spinner_item, ges_list);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);// looks better
        spinner.setAdapter(adapter);

        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int position, long id) {
                String selection = adapterView.getItemAtPosition(position).toString();

                if (selection.equals(ges_list[0])){
                    Toast.makeText(adapterView.getContext(), "Please choose a gesture", Toast.LENGTH_SHORT).show();
                }
                else{
                    Toast.makeText(adapterView.getContext(), "You chose: "+ selection, Toast.LENGTH_SHORT).show();
                    Intent intent2 = new Intent(MainActivity.this, Main2Activity.class);
                    intent2.putExtra("key", selection);
                    startActivity(intent2);
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {}
        });
    }
}
