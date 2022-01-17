package com.lauszus.facerecognitionapp.helper;

import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;

import com.lauszus.facerecognitionapp.NativeMethods;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

/**
 * author : HJY
 * date   : 2022/1/15 13:49
 * desc   :
 */
public class FaceRecognizeHelper {

    private static volatile FaceRecognizeHelper           instance;
    private static final    String                        TAG = FaceRecognizeHelper.class.getSimpleName();
    private                 NativeMethods.MeasureDistTask mMeasureDistTask;
    private                 NativeMethods.TrainFacesTask  mTrainFacesTask;
    private                 Mat                           mRgba, mGray;
    private boolean           useEigenfaces;
    private ArrayList<Mat>    images;
    private ArrayList<String> imagesLabels;
    private float             faceThreshold, distanceThreshold;
    private int                   maximumImages;
    private Context               mContext;
    private String[]              uniqueLabels;
    private LoadOpenCVCallBack    mLoadOpenCVCallBack;
    private FaceRecognizeCallBack mFaceRecognizeCallBack;

    public interface LoadOpenCVCallBack {
        void onLoadOpenCVSuccess();
    }

    public interface TrainFacesTipCallBack {
        void onHappenTip(int tipType, String tipMessage);
    }

    public interface FaceRecognizeCallBack {
        void onRecognizeResult(boolean success, String resultMessage);

        void onRecognizeError(int errorType, String errorMessage);
    }

    private FaceRecognizeHelper(Context context) {
        mContext = context;
    }

    public static FaceRecognizeHelper getInstance(Context context) {
        if (instance == null) {
            synchronized (FaceRecognizeHelper.class) {
                if (instance == null) {
                    instance = new FaceRecognizeHelper(context.getApplicationContext());
                }
            }
        }
        return instance;
    }


    public void initMat() {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void updateMat(Mat mGrayTmp, Mat mRgbaTmp) {
        mGray = mGrayTmp;
        mRgba = mRgbaTmp;
    }

    public Mat getRgba() {
        return mRgba;
    }

    public void releaseMat() {
        if (mGray != null) {
            mGray.release();
        }

        if (mRgba != null) {
            mRgba.release();
        }
    }


    public void initImagesAndLabel(ArrayList<Mat> images, ArrayList<String> imagesLabels) {
        this.images = images;
        this.imagesLabels = imagesLabels;
    }

    public ArrayList<Mat> getImages() {
        return images;
    }

    public ArrayList<String> getImagesLabels() {
        return imagesLabels;
    }

    public void loadOpenCV(LoadOpenCVCallBack loadOpenCVCallBack) {
        mLoadOpenCVCallBack = loadOpenCVCallBack;
        if (!OpenCVLoader.initDebug(true)) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, mContext, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }


    public void addLabel(TrainFacesTipCallBack trainFacesTipCallBack, String string) {
        String label = string.substring(0, 1).toUpperCase(Locale.US) + string.substring(1).trim().toLowerCase(Locale.US); // Make sure that the name is always uppercase and rest is lowercase
        imagesLabels.add(label); // Add label to list of labels
        Log.i(TAG, "Label: " + label);

        trainFaces(trainFacesTipCallBack); // When we have finished setting the label, then retrain faces
    }


    public void storeImagesAndLabel(SharedPreferences prefs, TinyDB tinydb, int mCameraIndex) {
        SharedPreferences.Editor editor = prefs.edit();
        editor.putFloat("faceThreshold", faceThreshold);
        editor.putFloat("distanceThreshold", distanceThreshold);
        editor.putInt("maximumImages", maximumImages);
        editor.putBoolean("useEigenfaces", useEigenfaces);
        editor.putInt("mCameraIndex", mCameraIndex);
        editor.apply();
        if (images != null && imagesLabels != null) {
            tinydb.putListMat("images", images);
            tinydb.putListString("imagesLabels", imagesLabels);
        }
    }


    public void clearImagesAndLabels() {
        images.clear(); // Clear both arrays, when new instance is created
        imagesLabels.clear();
    }

    public void setUseEigenfaces(boolean useEigenfaces) {
        this.useEigenfaces = useEigenfaces;
    }

    public boolean isUseEigenfaces() {
        return useEigenfaces;
    }

    public void faceRecognize(final Activity activity) {
        if (mGray == null) {
            return;
        }
        mFaceRecognizeCallBack = (FaceRecognizeCallBack) activity;
        if (mMeasureDistTask != null && mMeasureDistTask.getStatus() != AsyncTask.Status.FINISHED) {
            Log.i(TAG, "mMeasureDistTask is still running");
            activity.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    mFaceRecognizeCallBack.onRecognizeError(1, "Still processing old image...");
                }
            });
            return;
        }
        if (mTrainFacesTask != null && mTrainFacesTask.getStatus() != AsyncTask.Status.FINISHED) {
            Log.i(TAG, "mTrainFacesTask is still running");
            activity.runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    mFaceRecognizeCallBack.onRecognizeError(2, "Still training...");
                }
            });
            return;
        }

        Log.i(TAG, "Gray height: " + mGray.height() + " Width: " + mGray.width() + " total: " + mGray.total());
        if (mGray.total() == 0)
            return;

        // Scale image in order to decrease computation time and make the image square,
        // so it does not crash on phones with different aspect ratios for the front
        // and back camera
        Size imageSize = new Size(200, 200);
        Imgproc.resize(mGray, mGray, imageSize);
        Log.i(TAG, "Small gray height: " + mGray.height() + " Width: " + mGray.width() + " total: " + mGray.total());
        //SaveImage(mGray);

        Mat image = mGray.reshape(0, (int) mGray.total()); // Create column vector
        Log.i(TAG, "Vector height: " + image.height() + " Width: " + image.width() + " total: " + image.total());
        images.add(image); // Add current image to the array

        if (images.size() > maximumImages) {
            images.remove(0); // Remove first image
            if(!imagesLabels.isEmpty()){
                imagesLabels.remove(0); // Remove first label
            }
            Log.i(TAG, "The number of images is limited to: " + images.size());
        }

        // Calculate normalized Euclidean distance
        mMeasureDistTask = new NativeMethods.MeasureDistTask(useEigenfaces, measureDistTaskCallback);
        mMeasureDistTask.execute(image);
    }


    /**
     * Train faces using stored images.
     *
     * @return Returns false if the task is already running.
     */
    public boolean trainFaces(TrainFacesTipCallBack trainFacesTipCallBack) {
        if (trainFacesTipCallBack == null) {
            throw new RuntimeException("Must set trainFacesTipCallBack");
        }

        if (images.isEmpty())
            return true; // The array might be empty if the method is changed in the OnClickListener

        if (mTrainFacesTask != null && mTrainFacesTask.getStatus() != AsyncTask.Status.FINISHED) {
            Log.i(TAG, "mTrainFacesTask is still running");
            return false;
        }

        Mat imagesMatrix = new Mat((int) images.get(0).total(), images.size(), images.get(0).type());
        for (int i = 0; i < images.size(); i++)
            images.get(i).copyTo(imagesMatrix.col(i)); // Create matrix where each image is represented as a column vector

        Log.i(TAG, "Images height: " + imagesMatrix.height() + " Width: " + imagesMatrix.width() + " total: " + imagesMatrix.total());

        // Train the face recognition algorithms in an asynchronous task, so we do not skip any frames
        if (useEigenfaces) {
            Log.i(TAG, "Training Eigenfaces");
            trainFacesTipCallBack.onHappenTip(1, "Training Eigenfaces");

            mTrainFacesTask = new NativeMethods.TrainFacesTask(trainFacesTipCallBack, imagesMatrix, trainFacesTaskCallback);
        } else {
            Log.i(TAG, "Training Fisherfaces");
            trainFacesTipCallBack.onHappenTip(2, "Training Fisherfaces");

            Set<String> uniqueLabelsSet = new HashSet<>(imagesLabels); // Get all unique labels
            uniqueLabels = uniqueLabelsSet.toArray(new String[uniqueLabelsSet.size()]); // Convert to String array, so we can read the values from the indices

            int[] classesNumbers = new int[uniqueLabels.length];
            for (int i = 0; i < classesNumbers.length; i++)
                classesNumbers[i] = i + 1; // Create incrementing list for each unique label starting at 1

            int[] classes = new int[imagesLabels.size()];
            for (int i = 0; i < imagesLabels.size(); i++) {
                String label = imagesLabels.get(i);
                for (int j = 0; j < uniqueLabels.length; j++) {
                    if (label.equals(uniqueLabels[j])) {
                        classes[i] = classesNumbers[j]; // Insert corresponding number
                        break;
                    }
                }
            }

            /*for (int i = 0; i < imagesLabels.size(); i++)
                Log.i(TAG, "Classes: " + imagesLabels.get(i) + " = " + classes[i]);*/

            Mat vectorClasses = new Mat(classes.length, 1, CvType.CV_32S); // CV_32S == int
            vectorClasses.put(0, 0, classes); // Copy int array into a vector

            mTrainFacesTask = new NativeMethods.TrainFacesTask(trainFacesTipCallBack, imagesMatrix, vectorClasses, trainFacesTaskCallback);
        }
        mTrainFacesTask.execute();

        return true;
    }


    public void setMaximumImages(int maximumImages) {
        this.maximumImages = maximumImages;
    }

    public int getMaximumImages() {
        return maximumImages;
    }


    public float getFaceThreshold() {
        return faceThreshold;
    }

    public void setFaceThreshold(float faceThreshold) {
        this.faceThreshold = faceThreshold;
    }

    public float getDistanceThreshold() {
        return distanceThreshold;
    }

    public void setDistanceThreshold(float distanceThreshold) {
        this.distanceThreshold = distanceThreshold;
    }

    public void updateMaximumImagesAndRetrainFaces(TrainFacesTipCallBack trainFacesTipCallBack, int num) {
        maximumImages = num;
        if (images != null && images.size() > maximumImages) {
            int nrRemoveImages = images.size() - maximumImages;
            Log.i(TAG, "Removed " + nrRemoveImages + " images from the list");
            images.subList(0, nrRemoveImages).clear(); // Remove oldest images
            imagesLabels.subList(0, nrRemoveImages).clear(); // Remove oldest labels
            trainFaces(trainFacesTipCallBack); // Retrain faces
        }
    }


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(mContext) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    NativeMethods.loadNativeLibraries(); // Load native libraries after(!) OpenCV initialization
                    Log.i(TAG, "OpenCV loaded successfully");
                    if (mLoadOpenCVCallBack != null) {
                        mLoadOpenCVCallBack.onLoadOpenCVSuccess();
                    }
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };


    private NativeMethods.TrainFacesTask.Callback trainFacesTaskCallback = new NativeMethods.TrainFacesTask.Callback() {
        @Override
        public void onTrainFacesComplete(boolean result, TrainFacesTipCallBack trainFacesTipCallBack) {
            if (result)
                trainFacesTipCallBack.onHappenTip(3, "Training complete");
            else
                trainFacesTipCallBack.onHappenTip(4, "Training failed");
        }
    };


    private NativeMethods.MeasureDistTask.Callback measureDistTaskCallback = new NativeMethods.MeasureDistTask.Callback() {
        @Override
        public void onMeasureDistComplete(Bundle bundle) {
            if (bundle == null) {
                if (mFaceRecognizeCallBack != null) {
                    mFaceRecognizeCallBack.onRecognizeResult(false, "Failed to measure distance");
                }
                return;
            }

            float minDist = bundle.getFloat(NativeMethods.MeasureDistTask.MIN_DIST_FLOAT);
            boolean recognizeSuccess = false;
            String resultMsg = null;
            if (minDist != -1) {
                int minIndex = bundle.getInt(NativeMethods.MeasureDistTask.MIN_DIST_INDEX_INT);
                float faceDist = bundle.getFloat(NativeMethods.MeasureDistTask.DIST_FACE_FLOAT);
                if (imagesLabels.size() > minIndex) { // Just to be sure
                    Log.i(TAG, "dist[" + minIndex + "]: " + minDist + ", face dist: " + faceDist + ", label: " + imagesLabels.get(minIndex));

                    String minDistString = String.format(Locale.US, "%.4f", minDist);
                    String faceDistString = String.format(Locale.US, "%.4f", faceDist);

                    if (faceDist < faceThreshold && minDist < distanceThreshold) {// 1. Near face space and near a face class
                        recognizeSuccess = true;
                        resultMsg = "Face detected: " + imagesLabels.get(minIndex) + ". Distance: " + minDistString;
                    } else if (faceDist < faceThreshold) { // 2. Near face space but not near a known face class
                        resultMsg = "Unknown face. Face distance: " + faceDistString + ". Closest Distance: " + minDistString;
                    } else if (minDist < distanceThreshold) { // 3. Distant from face space and near a face class
                        resultMsg = "False recognition. Face distance: " + faceDistString + ". Closest Distance: " + minDistString;
                    } else { // 4. Distant from face space and not near a known face class.
                        resultMsg = "Image is not a face. Face distance: " + faceDistString + ". Closest Distance: " + minDistString;
                    }
                }
            } else {
                Log.w(TAG, "Array is null");
                if (useEigenfaces || uniqueLabels == null || uniqueLabels.length > 1)
                    resultMsg = "Keep training...";
                else
                    resultMsg = "Fisherfaces needs two different faces";
            }

            if (mFaceRecognizeCallBack != null) {
                mFaceRecognizeCallBack.onRecognizeResult(recognizeSuccess, resultMsg);
            }

        }
    };

}
