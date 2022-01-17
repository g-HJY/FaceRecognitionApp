/*******************************************************************************
 * Copyright (C) 2016 Kristian Sloth Lauszus. All rights reserved.
 *
 * This software may be distributed and modified under the terms of the GNU
 * General Public License version 2 (GPL2) as published by the Free Software
 * Foundation and appearing in the file GPL2.TXT included in the packaging of
 * this file. Please note that GPL2 Section 2[b] requires that all works based
 * on this software must also be made publicly available under the terms of
 * the GPL2 ("Copyleft").
 *
 * Contact information
 * -------------------
 *
 * Kristian Sloth Lauszus
 * Web      :  http://www.lauszus.com
 * e-mail   :  lauszus@gmail.com
 ******************************************************************************/

package com.lauszus.facerecognitionapp.activity;

import android.Manifest;
import android.animation.Animator;
import android.animation.ObjectAnimator;
import android.app.Activity;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.preference.PreferenceManager;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v4.view.GravityCompat;
import android.support.v4.widget.DrawerLayout;
import android.support.v7.app.ActionBarDrawerToggle;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.text.InputType;
import android.util.Log;
import android.view.GestureDetector;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.RadioButton;
import android.widget.TextView;
import android.widget.Toast;

import com.lauszus.facerecognitionapp.R;
import com.lauszus.facerecognitionapp.helper.FaceRecognizeHelper;
import com.lauszus.facerecognitionapp.helper.TinyDB;
import com.lauszus.facerecognitionapp.view.CameraBridgeViewBase;
import com.lauszus.facerecognitionapp.view.SeekBarArrows;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

public class FaceRecognitionAppActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2,
        FaceRecognizeHelper.LoadOpenCVCallBack, FaceRecognizeHelper.FaceRecognizeCallBack {
    private static final String TAG                      = FaceRecognitionAppActivity.class.getSimpleName();
    private static final int    PERMISSIONS_REQUEST_CODE = 0;

    private CameraBridgeViewBase mOpenCvCameraView;

    private SeekBarArrows mThresholdFace, mThresholdDistance, mMaximumImages;
    private SharedPreferences prefs;
    private TinyDB            tinydb;
    private Toolbar           mToolbar;
    private Toast             mToast;
    private FaceRecognizeHelper mFaceRecognizeHelper;


    private void showLabelsDialog() {
        Set<String> uniqueLabelsSet = new HashSet<String>(mFaceRecognizeHelper.getImagesLabels()); // Get all unique labels
        if (!uniqueLabelsSet.isEmpty()) { // Make sure that there are any labels
            // Inspired by: http://stackoverflow.com/questions/15762905/how-can-i-display-a-list-view-in-an-android-alert-dialog
            AlertDialog.Builder builder = new AlertDialog.Builder(FaceRecognitionAppActivity.this);
            builder.setTitle("Select label:");
            builder.setPositiveButton("New face", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    dialog.dismiss();
                    showEnterLabelDialog();
                }
            });
            builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
                @Override
                public void onClick(DialogInterface dialog, int which) {
                    dialog.dismiss();
                    //images.remove(images.size() - 1); // Remove last image
                }
            });
            builder.setCancelable(false); // Prevent the user from closing the dialog

            String[] uniqueLabels = uniqueLabelsSet.toArray(new String[uniqueLabelsSet.size()]); // Convert to String array for ArrayAdapter
            Arrays.sort(uniqueLabels); // Sort labels alphabetically
            final ArrayAdapter<String> arrayAdapter = new ArrayAdapter<String>(FaceRecognitionAppActivity.this, android.R.layout.simple_list_item_1, uniqueLabels) {
                @Override
                public @NonNull
                View getView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {
                    TextView textView = (TextView) super.getView(position, convertView, parent);
                    if (getResources().getBoolean(R.bool.isTablet))
                        textView.setTextSize(20); // Make text slightly bigger on tablets compared to phones
                    else
                        textView.setTextSize(18); // Increase text size a little bit
                    return textView;
                }
            };
            ListView mListView = new ListView(FaceRecognitionAppActivity.this);
            mListView.setAdapter(arrayAdapter); // Set adapter, so the items actually show up
            builder.setView(mListView); // Set the ListView

            final AlertDialog dialog = builder.show(); // Show dialog and store in final variable, so it can be dismissed by the ListView

            mListView.setOnItemClickListener(new AdapterView.OnItemClickListener() {
                @Override
                public void onItemClick(AdapterView<?> parent, View view, int position, long id) {
                    dialog.dismiss();
                    mFaceRecognizeHelper.addLabel(new FaceRecognizeHelper.TrainFacesTipCallBack() {
                        @Override
                        public void onHappenTip(int tipType, String tipMessage) {
                           showToast(FaceRecognitionAppActivity.this,tipMessage,Toast.LENGTH_LONG);
                        }
                    }, arrayAdapter.getItem(position));
                }
            });
        } else
            showEnterLabelDialog(); // If there is no existing labels, then ask the user for a new label
    }

    private void showEnterLabelDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(FaceRecognitionAppActivity.this);
        builder.setTitle("Please enter your name:");

        final EditText input = new EditText(FaceRecognitionAppActivity.this);
        input.setInputType(InputType.TYPE_CLASS_TEXT);
        builder.setView(input);

        builder.setPositiveButton("Submit", null); // Set up positive button, but do not provide a listener, so we can check the string before dismissing the dialog
        builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                dialog.dismiss();
                //images.remove(images.size() - 1); // Remove last image
            }
        });
        builder.setCancelable(false); // User has to input a name
        AlertDialog dialog = builder.create();

        // Source: http://stackoverflow.com/a/7636468/2175837
        dialog.setOnShowListener(new DialogInterface.OnShowListener() {
            @Override
            public void onShow(final DialogInterface dialog) {
                Button mButton = ((AlertDialog) dialog).getButton(AlertDialog.BUTTON_POSITIVE);
                mButton.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        String string = input.getText().toString().trim();
                        if (!string.isEmpty()) { // Make sure the input is valid
                            // If input is valid, dismiss the dialog and add the label to the array
                            dialog.dismiss();
                            mFaceRecognizeHelper.addLabel(new FaceRecognizeHelper.TrainFacesTipCallBack() {
                                @Override
                                public void onHappenTip(int tipType, String tipMessage) {
                                    showToast(FaceRecognitionAppActivity.this,tipMessage,Toast.LENGTH_LONG);
                                }
                            }, string);
                        }
                    }
                });
            }
        });

        // Show keyboard, so the user can start typing straight away
        dialog.getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_VISIBLE);

        dialog.show();
    }


    @Override
    public void onLoadOpenCVSuccess() {
        mOpenCvCameraView.enableView();

        // Read images and labels from shared preferences
        mFaceRecognizeHelper.initImagesAndLabel(tinydb.getListMat("images"),tinydb.getListString("imagesLabels"));
        List<Mat> images = mFaceRecognizeHelper.getImages();
        List<String> imagesLabels = mFaceRecognizeHelper.getImagesLabels();

        Log.i(TAG, "Number of images: " + images.size() + ". Number of labels: " + imagesLabels.size());
        if (!images.isEmpty()) {
            mFaceRecognizeHelper.trainFaces(new FaceRecognizeHelper.TrainFacesTipCallBack() {
                @Override
                public void onHappenTip(int tipType, String tipMessage) {
                    showToast(FaceRecognitionAppActivity.this,tipMessage,Toast.LENGTH_LONG);
                }
            }); // Train images after they are loaded
            Log.i(TAG, "Images height: " + images.get(0).height() + " Width: " + images.get(0).width() + " total: " + images.get(0).total());
        }
        Log.i(TAG, "Labels: " + imagesLabels);
    }


    @Override
    public void onRecognizeResult(boolean success, String resultMessage) {
        if (success) {
            //人脸识别成功
        }
        showToast(FaceRecognitionAppActivity.this, resultMessage, Toast.LENGTH_LONG);
    }

    @Override
    public void onRecognizeError(int errorType, String errorMessage) {
        showToast(FaceRecognitionAppActivity.this, errorMessage, Toast.LENGTH_LONG);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_face_recognition_app);
        mToolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(mToolbar); // Sets the Toolbar to act as the ActionBar for this Activity window

        DrawerLayout drawer = (DrawerLayout) findViewById(R.id.drawer_layout);
        ActionBarDrawerToggle toggle = new ActionBarDrawerToggle(this, drawer, mToolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close);
        drawer.addDrawerListener(toggle);
        toggle.syncState();

        //初始化人脸识别辅助类
        mFaceRecognizeHelper = FaceRecognizeHelper.getInstance(FaceRecognitionAppActivity.this);


        final RadioButton mRadioButtonEigenfaces = (RadioButton) findViewById(R.id.eigenfaces);
        final RadioButton mRadioButtonFisherfaces = (RadioButton) findViewById(R.id.fisherfaces);

        mRadioButtonEigenfaces.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mFaceRecognizeHelper.setUseEigenfaces(true);
                if (!mFaceRecognizeHelper.trainFaces(new FaceRecognizeHelper.TrainFacesTipCallBack() {
                    @Override
                    public void onHappenTip(int tipType, String tipMessage) {
                        showToast(FaceRecognitionAppActivity.this,tipMessage,Toast.LENGTH_LONG);
                    }
                })) {
                    mFaceRecognizeHelper.setUseEigenfaces(false); // Set variable back
                    showToast(FaceRecognitionAppActivity.this, "Still training...", Toast.LENGTH_SHORT);
                    mRadioButtonEigenfaces.setChecked(mFaceRecognizeHelper.isUseEigenfaces());
                    mRadioButtonFisherfaces.setChecked(!mFaceRecognizeHelper.isUseEigenfaces());
                }
            }
        });
        mRadioButtonFisherfaces.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mFaceRecognizeHelper.setUseEigenfaces(false);
                if (!mFaceRecognizeHelper.trainFaces(new FaceRecognizeHelper.TrainFacesTipCallBack() {
                    @Override
                    public void onHappenTip(int tipType, String tipMessage) {
                        showToast(FaceRecognitionAppActivity.this,tipMessage,Toast.LENGTH_LONG);
                    }
                })) {
                    mFaceRecognizeHelper.setUseEigenfaces(true); // Set variable back
                    showToast(FaceRecognitionAppActivity.this, "Still training...", Toast.LENGTH_SHORT);
                    mRadioButtonEigenfaces.setChecked(mFaceRecognizeHelper.isUseEigenfaces());
                    mRadioButtonFisherfaces.setChecked(!mFaceRecognizeHelper.isUseEigenfaces());
                }
            }
        });

        // Set radio button based on value stored in shared preferences
        prefs = PreferenceManager.getDefaultSharedPreferences(this);
        mFaceRecognizeHelper.setUseEigenfaces(prefs.getBoolean("useEigenfaces", true));
        mRadioButtonEigenfaces.setChecked(mFaceRecognizeHelper.isUseEigenfaces());
        mRadioButtonFisherfaces.setChecked(!mFaceRecognizeHelper.isUseEigenfaces());

        tinydb = new TinyDB(this); // Used to store ArrayLists in the shared preferences

        mThresholdFace = (SeekBarArrows) findViewById(R.id.threshold_face);
        mThresholdFace.setOnSeekBarArrowsChangeListener(new SeekBarArrows.OnSeekBarArrowsChangeListener() {
            @Override
            public void onProgressChanged(float progress) {
                Log.i(TAG, "Face threshold: " + mThresholdFace.progressToString(progress));
                mFaceRecognizeHelper.setFaceThreshold(progress);
            }
        });
        mFaceRecognizeHelper.setFaceThreshold(mThresholdFace.getProgress());

        mThresholdDistance = (SeekBarArrows) findViewById(R.id.threshold_distance);
        mThresholdDistance.setOnSeekBarArrowsChangeListener(new SeekBarArrows.OnSeekBarArrowsChangeListener() {
            @Override
            public void onProgressChanged(float progress) {
                Log.i(TAG, "Distance threshold: " + mThresholdDistance.progressToString(progress));
                mFaceRecognizeHelper.setDistanceThreshold(progress);
            }
        });

        mFaceRecognizeHelper.setDistanceThreshold(mThresholdDistance.getProgress());// Get initial value

        mMaximumImages = (SeekBarArrows) findViewById(R.id.maximum_images);
        mMaximumImages.setOnSeekBarArrowsChangeListener(new SeekBarArrows.OnSeekBarArrowsChangeListener() {
            @Override
            public void onProgressChanged(float progress) {
                Log.i(TAG, "Maximum number of images: " + mMaximumImages.progressToString(progress));
                mFaceRecognizeHelper.updateMaximumImagesAndRetrainFaces(new FaceRecognizeHelper.TrainFacesTipCallBack() {
                    @Override
                    public void onHappenTip(int tipType, String tipMessage) {
                        showToast(FaceRecognitionAppActivity.this,tipMessage,Toast.LENGTH_LONG);
                    }
                }, (int) progress);
            }
        });


        mFaceRecognizeHelper.setMaximumImages((int) mMaximumImages.getProgress()); // Get initial value

        findViewById(R.id.clear_button).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.i(TAG, "Cleared training set");
                mFaceRecognizeHelper.clearImagesAndLabels();
                showToast(FaceRecognitionAppActivity.this, "Training set cleared", Toast.LENGTH_SHORT);
            }
        });

        findViewById(R.id.take_picture_button).setOnClickListener(new View.OnClickListener() {


            @Override
            public void onClick(View v) {

                mFaceRecognizeHelper.faceRecognize(FaceRecognitionAppActivity.this);
                showLabelsDialog();
            }
        });


        final GestureDetector mGestureDetector = new GestureDetector(this, new GestureDetector.SimpleOnGestureListener() {
            @Override
            public boolean onDown(MotionEvent e) {
                return true;
            }

            @Override
            public boolean onDoubleTap(MotionEvent e) {
                // Show flip animation when the camera is flipped due to a double tap
                flipCameraAnimation();
                return true;
            }
        });

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_java_surface_view);
        mOpenCvCameraView.setCameraIndex(prefs.getInt("mCameraIndex", CameraBridgeViewBase.CAMERA_ID_FRONT));
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                return mGestureDetector.onTouchEvent(event);
            }
        });
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String permissions[], @NonNull int[] grantResults) {
        switch (requestCode) {
            case PERMISSIONS_REQUEST_CODE:
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    mFaceRecognizeHelper.loadOpenCV(this);
                } else {
                    showToast(FaceRecognitionAppActivity.this, "Permission required!", Toast.LENGTH_LONG);
                    finish();
                }
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onStart() {
        super.onStart();
        // Read threshold values
        float progress = prefs.getFloat("faceThreshold", -1);
        if (progress != -1)
            mThresholdFace.setProgress(progress);
        progress = prefs.getFloat("distanceThreshold", -1);
        if (progress != -1)
            mThresholdDistance.setProgress(progress);
        mMaximumImages.setProgress(prefs.getInt("maximumImages", 25)); // Use 25 images by default
    }

    @Override
    public void onStop() {
        super.onStop();
        // Store threshold values
        // Store ArrayLists containing the images and labels
        mFaceRecognizeHelper.storeImagesAndLabel(prefs,tinydb,mOpenCvCameraView.mCameraIndex);
    }

    @Override
    public void onResume() {
        super.onResume();

        // Request permission if needed
        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED/* || ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED*/)
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA/*, Manifest.permission.WRITE_EXTERNAL_STORAGE*/}, PERMISSIONS_REQUEST_CODE);
        else
            mFaceRecognizeHelper.loadOpenCV(this);
    }


    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mFaceRecognizeHelper.initMat();
    }

    public void onCameraViewStopped() {
        mFaceRecognizeHelper.releaseMat();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat mGrayTmp = inputFrame.gray();
        Mat mRgbaTmp = inputFrame.rgba();

        // Flip image to get mirror effect
        int orientation = mOpenCvCameraView.getScreenOrientation();
        if (mOpenCvCameraView.isEmulator()) // Treat emulators as a special case
            Core.flip(mRgbaTmp, mRgbaTmp, 1); // Flip along y-axis
        else {
            switch (orientation) { // RGB image
                case ActivityInfo.SCREEN_ORIENTATION_PORTRAIT:
                case ActivityInfo.SCREEN_ORIENTATION_REVERSE_PORTRAIT:
                    if (mOpenCvCameraView.mCameraIndex == CameraBridgeViewBase.CAMERA_ID_FRONT)
                        Core.flip(mRgbaTmp, mRgbaTmp, 0); // Flip along x-axis
                    else
                        Core.flip(mRgbaTmp, mRgbaTmp, -1); // Flip along both axis
                    break;
                case ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE:
                case ActivityInfo.SCREEN_ORIENTATION_REVERSE_LANDSCAPE:
                    if (mOpenCvCameraView.mCameraIndex == CameraBridgeViewBase.CAMERA_ID_FRONT)
                        Core.flip(mRgbaTmp, mRgbaTmp, 1); // Flip along y-axis
                    break;
            }
            switch (orientation) { // Grayscale image
                case ActivityInfo.SCREEN_ORIENTATION_PORTRAIT:
                    Core.transpose(mGrayTmp, mGrayTmp); // Rotate image
                    if (mOpenCvCameraView.mCameraIndex == CameraBridgeViewBase.CAMERA_ID_FRONT)
                        Core.flip(mGrayTmp, mGrayTmp, -1); // Flip along both axis
                    else
                        Core.flip(mGrayTmp, mGrayTmp, 1); // Flip along y-axis
                    break;
                case ActivityInfo.SCREEN_ORIENTATION_REVERSE_PORTRAIT:
                    Core.transpose(mGrayTmp, mGrayTmp); // Rotate image
                    if (mOpenCvCameraView.mCameraIndex == CameraBridgeViewBase.CAMERA_ID_BACK)
                        Core.flip(mGrayTmp, mGrayTmp, 0); // Flip along x-axis
                    break;
                case ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE:
                    if (mOpenCvCameraView.mCameraIndex == CameraBridgeViewBase.CAMERA_ID_FRONT)
                        Core.flip(mGrayTmp, mGrayTmp, 1); // Flip along y-axis
                    break;
                case ActivityInfo.SCREEN_ORIENTATION_REVERSE_LANDSCAPE:
                    Core.flip(mGrayTmp, mGrayTmp, 0); // Flip along x-axis
                    if (mOpenCvCameraView.mCameraIndex == CameraBridgeViewBase.CAMERA_ID_BACK)
                        Core.flip(mGrayTmp, mGrayTmp, 1); // Flip along y-axis
                    break;
            }
        }

        mFaceRecognizeHelper.updateMat(mGrayTmp, mRgbaTmp);


        return mFaceRecognizeHelper.getRgba();
    }

    @SuppressWarnings("ResultOfMethodCallIgnored")
    public void SaveImage(Mat mat) {
        Mat mIntermediateMat = new Mat();

        if (mat.channels() == 1) // Grayscale image
            Imgproc.cvtColor(mat, mIntermediateMat, Imgproc.COLOR_GRAY2BGR);
        else
            Imgproc.cvtColor(mat, mIntermediateMat, Imgproc.COLOR_RGBA2BGR);

        File path = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), TAG); // Save pictures in Pictures directory
        path.mkdir(); // Create directory if needed
        String fileName = "IMG_" + new SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(new Date()) + ".png";
        File file = new File(path, fileName);

        boolean bool = Imgcodecs.imwrite(file.toString(), mIntermediateMat);

        if (bool)
            Log.i(TAG, "SUCCESS writing image to external storage");
        else
            Log.e(TAG, "Failed writing image to external storage");
    }

    @Override
    public void onBackPressed() {
        DrawerLayout drawer = (DrawerLayout) findViewById(R.id.drawer_layout);
        if (drawer.isDrawerOpen(GravityCompat.START))
            drawer.closeDrawer(GravityCompat.START);
        else
            super.onBackPressed();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_face_recognition_app, menu);
        // Show rear camera icon if front camera is currently used and front camera icon if back camera is used
        MenuItem menuItem = menu.findItem(R.id.flip_camera);
        if (mOpenCvCameraView.mCameraIndex == CameraBridgeViewBase.CAMERA_ID_FRONT)
            menuItem.setIcon(R.drawable.ic_camera_front_white_24dp);
        else
            menuItem.setIcon(R.drawable.ic_camera_rear_white_24dp);
        return true;
    }

    private void flipCameraAnimation() {
        // Flip the camera
        mOpenCvCameraView.flipCamera();

        // Do flip camera animation
        View v = mToolbar.findViewById(R.id.flip_camera);
        ObjectAnimator animator = ObjectAnimator.ofFloat(v, "rotationY", v.getRotationY() + 180.0f);
        animator.setDuration(500);
        animator.addListener(new Animator.AnimatorListener() {
            @Override
            public void onAnimationStart(Animator animation) {

            }

            @Override
            public void onAnimationEnd(Animator animation) {
                supportInvalidateOptionsMenu(); // This will call onCreateOptionsMenu()
            }

            @Override
            public void onAnimationCancel(Animator animation) {

            }

            @Override
            public void onAnimationRepeat(Animator animation) {

            }
        });
        animator.start();
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.flip_camera:
                flipCameraAnimation();
                return true;
        }
        return super.onOptionsItemSelected(item);
    }


    public void showToast(final Activity activity, final String message, final int duration) {
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (duration != Toast.LENGTH_SHORT && duration != Toast.LENGTH_LONG)
                    throw new IllegalArgumentException();
                if (mToast != null && mToast.getView().isShown())
                    mToast.cancel(); // Close the toast if it is already open
                mToast = Toast.makeText(activity, message, duration);
                mToast.show();
            }
        });
    }

}
