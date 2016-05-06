package com.jnardari.opencv_androidsamples.activities;

import android.app.Activity;
import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.WindowManager;

import com.jnardari.opencv_androidsamples.R;
import com.jnardari.opencv_androidsamples.utils.ColorBlobDetector;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class MyTestActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "MyTestActivity";
    private CameraBridgeViewBase vOpenCvCameraView;
    int thresh = 50, N = 5; // karlphillip: decreased N to 2, was 11.
    List<MatOfPoint> squares = new ArrayList<MatOfPoint>();
    private Mat rgba;
    private List<MatOfPoint> toDraw = new ArrayList<>();
    private ColorBlobDetector mDetector;


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    vOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };
    private Scalar mBlobColorHsv;
    private Scalar mBlobColorRgba;
    private Mat mSpectrum;
    private Size SPECTRUM_SIZE;
    private Scalar CONTOUR_COLOR;
    private boolean mIsColorSelected = false;
    private Mat rgbaInnerWindow;
    private Mat mIntermediateMat;
    private double mMaxCurveLength = 0;
    private boolean isCaculating = false;


    public static void launch(Activity activity) {
        activity.startActivity(new Intent(activity, MyTestActivity.class));
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_find_rectangle);
        initViews();
    }

    private void initViews() {
        vOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera);
        vOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (vOpenCvCameraView != null)
            vOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (vOpenCvCameraView != null)
            vOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mIntermediateMat = new Mat();
        rgba = new Mat(height, width, CvType.CV_8UC4);
        mDetector = new ColorBlobDetector();
        mSpectrum = new Mat();
        mBlobColorRgba = new Scalar(255);
        mBlobColorHsv = new Scalar(255);
        SPECTRUM_SIZE = new Size(200, 64);
        CONTOUR_COLOR = new Scalar(255, 0, 0, 255);
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(final CameraBridgeViewBase.CvCameraViewFrame inputFrame) {


        if (mMaxCurveLength == 0) {
            mMaxCurveLength = inputFrame.rgba().width() * 2 + inputFrame.rgba().height() * 2;
        }
        if (!isCaculating) {
            new AsyncTask<Void, Void, List<MatOfPoint>>(){

                @Override
                protected List<MatOfPoint> doInBackground(Void... params) {
                    return findSquares(inputFrame.rgba());
                }

                @Override
                protected void onPreExecute() {
                    super.onPreExecute();
                    isCaculating = true;
                }

                @Override
                protected void onPostExecute(List<MatOfPoint> filteredContours) {
                    super.onPostExecute(filteredContours);
                    isCaculating = false;
                    synchronized (squares) {

                        Log.d("xunqun", "squares found: " + filteredContours.size());
                        squares = new ArrayList<MatOfPoint>(filteredContours);
                    }

                }
            }.execute();
        }

        Mat image = inputFrame.rgba();

        synchronized (squares) {
            Imgproc.drawContours(image, squares, -1, new Scalar(255, 0, 0));
        }

        return image;
    }

    // helper function:
    // finds a cosine of angle between vectors
    // from pt0->pt1 and from pt0->pt2

    double angle(Point pt1, Point pt2, Point pt0) {
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return (dx1 * dx2 + dy1 * dy2) / Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10);
    }

    // returns sequence of squares detected on the image.
    // the sequence is stored in the specified memory storage
    // returns sequence of squares detected on the image.
    // the sequence is stored in the specified memory storage
    List<MatOfPoint> findSquares(Mat src) {
        Mat image = new Mat();
        Imgproc.cvtColor(src, image, Imgproc.COLOR_RGB2HSV, 4);

        List<MatOfPoint> filteredContours = new ArrayList<>();

        Mat smallerImg = new Mat(new Size(image.width() / 2, image.height() / 2), image.type());

        Mat gray = new Mat(image.size(), image.type());

        Mat gray0 = new Mat(image.size(), CvType.CV_8U);

        // down-scale and upscale the image to filter out the noise
        Imgproc.pyrDown(image, smallerImg, smallerImg.size());
        Imgproc.pyrUp(smallerImg, image, image.size());

        // find squares in every color plane of the image
        for (int c = 0; c < 3; c++) {

            extractChannel(image, gray, c);

            // try several threshold levels
            for (int l = 1; l < N; l++) {
                //Cany removed... Didn't work so well


                Imgproc.threshold(gray, gray0, (l + 1) * 255 / N, 255, Imgproc.THRESH_BINARY);

                List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

                // find contours and store them all as a list
                Imgproc.findContours(gray0, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

                if (contours.size() > 50) continue;
                // test each contour


                // caculation
                for (int i = 0; i < contours.size(); i++) {

                    // approximate contour with accuracy proportional
                    // to the contour perimeter

                    double arcLength = Imgproc.arcLength(new MatOfPoint2f(contours.get(i).toArray()), true);

                    if(arcLength < mMaxCurveLength * 0.2 || arcLength > mMaxCurveLength * 0.95)continue;

                    MatOfPoint approx = new MatOfPoint();

                    approx = approxPolyDP(contours.get(i), Imgproc.arcLength(new MatOfPoint2f(contours.get(i).toArray()), true) * 0.02, true);


                    // square contours should have 4 vertices after approximation
                    // relatively large area (to filter out noisy contours)
                    // and be convex.
                    // Note: absolute value of an area is used because
                    // area may be positive or negative - in accordance with the
                    // contour orientation

                    if (approx.toArray().length == 4 &&
                            Math.abs(Imgproc.contourArea(approx)) > 1000 &&
                            Imgproc.isContourConvex(approx)) {
                        double maxCosine = 0;

                        for (int j = 2; j < 5; j++) {
                            // find the maximum cosine of the angle between joint edges
                            double cosine = Math.abs(angle(approx.toArray()[j % 4], approx.toArray()[j - 2], approx.toArray()[j - 1]));
                            maxCosine = Math.max(maxCosine, cosine);
                        }

                        // if cosines of all angles are small
                        // (all angles are ~90 degree) then write quandrange
                        // vertices to resultant sequence
                        if (maxCosine < 0.3)
                            filteredContours.add(approx);
                    }
                }

            }
        }
        return filteredContours;
    }


    void extractChannel(Mat source, Mat out, int channelNum) {
        List<Mat> sourceChannels = new ArrayList<Mat>();
        List<Mat> outChannel = new ArrayList<Mat>();

        Core.split(source, sourceChannels);

        outChannel.add(new Mat(sourceChannels.get(0).size(), sourceChannels.get(0).type()));

        Core.mixChannels(sourceChannels, outChannel, new MatOfInt(channelNum, 0));

        Core.merge(outChannel, out);
    }

    MatOfPoint approxPolyDP(MatOfPoint curve, double epsilon, boolean closed) {
        MatOfPoint2f tempMat = new MatOfPoint2f();

        Imgproc.approxPolyDP(new MatOfPoint2f(curve.toArray()), tempMat, epsilon, closed);

        return new MatOfPoint(tempMat.toArray());
    }


    private Scalar converScalarHsv2Rgba(Scalar hsvColor) {
        Mat pointMatRgba = new Mat();
        Mat pointMatHsv = new Mat(1, 1, CvType.CV_8UC3, hsvColor);
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);

        return new Scalar(pointMatRgba.get(0, 0));
    }
}
