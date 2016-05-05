package com.jnardari.opencv_androidsamples.activities;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.view.View;

import com.jnardari.opencv_androidsamples.R;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class SamplesActivity extends AppCompatActivity {

    private static final int MY_PERMISSIONS_REQUEST_CAMERA = 123;
    Intent sampleIntent;
    int thresh = 50, N = 11;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_samples);

    }

    @Override
    protected void onResume() {
        super.onResume();
        checkPremission();
        
    }

    private void checkPremission() {
        if (ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {

            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA},
                    MY_PERMISSIONS_REQUEST_CAMERA);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String permissions[], int[] grantResults) {
        switch (requestCode) {
            case MY_PERMISSIONS_REQUEST_CAMERA: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

                } else {
                    finish();
                }
                return;
            }

            // other 'case' lines to check for other
            // permissions this app might request
        }
    }

    public void tutorial1(View v) {
        sampleIntent = new Intent(this, Tutorial1Activity.class);
        startActivity(sampleIntent);
    }

    public void tutorial2(View v) {
        sampleIntent = new Intent(this, Tutorial2Activity.class);
        startActivity(sampleIntent);
    }

    public void tutorial3(View v) {
        sampleIntent = new Intent(this, Tutorial3Activity.class);
        startActivity(sampleIntent);
    }

    public void imageManipulations(View v) {
        sampleIntent = new Intent(this, ImageManipulationsActivity.class);
        startActivity(sampleIntent);
    }

    public void faceDetection(View v) {
        sampleIntent = new Intent(this, FaceDetectionActivity.class);
        startActivity(sampleIntent);
    }

    public void colorBlobDetection(View v) {
        sampleIntent = new Intent(this, ColorBlobDetectionActivity.class);
        startActivity(sampleIntent);
    }

    public void cameraCalibration(View v) {
        sampleIntent = new Intent(this, CameraCalibrationActivity.class);
        startActivity(sampleIntent);
    }

    public void puzzle15(View v) {
        sampleIntent = new Intent(this, Puzzle15Activity.class);
        startActivity(sampleIntent);
    }

    public void mytest(View v){
        MyTestActivity.launch(this);
    }

    // returns sequence of squares detected on the image.
    // the sequence is stored in the specified memory storage
    void findSquares( Mat image, List<MatOfPoint> squares )
    {

        squares.clear();

        Mat smallerImg=new Mat(new Size(image.width()/2, image.height()/2),image.type());

        Mat gray=new Mat(image.size(),image.type());

        Mat gray0=new Mat(image.size(), CvType.CV_8U);

        // down-scale and upscale the image to filter out the noise
        Imgproc.pyrDown(image, smallerImg, smallerImg.size());
        Imgproc.pyrUp(smallerImg, image, image.size());

        // find squares in every color plane of the image
        for( int c = 0; c < 3; c++ )
        {

            extractChannel(image, gray, c);

            // try several threshold levels
            for( int l = 1; l < N; l++ )
            {
                //Cany removed... Didn't work so well


                Imgproc.threshold(gray, gray0, (l+1)*255/N, 255, Imgproc.THRESH_BINARY);


                List<MatOfPoint> contours=new ArrayList<MatOfPoint>();

                // find contours and store them all as a list
                Imgproc.findContours(gray0, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

                MatOfPoint approx=new MatOfPoint();

                // test each contour
                for( int i = 0; i < contours.size(); i++ )
                {

                    // approximate contour with accuracy proportional
                    // to the contour perimeter
                    approx = approxPolyDP(contours.get(i),  Imgproc.arcLength(new MatOfPoint2f(contours.get(i).toArray()), true)*0.02, true);


                    // square contours should have 4 vertices after approximation
                    // relatively large area (to filter out noisy contours)
                    // and be convex.
                    // Note: absolute value of an area is used because
                    // area may be positive or negative - in accordance with the
                    // contour orientation

                    if( approx.toArray().length == 4 &&
                            Math.abs(Imgproc.contourArea(approx)) > 1000 &&
                            Imgproc.isContourConvex(approx) )
                    {
                        double maxCosine = 0;

                        for( int j = 2; j < 5; j++ )
                        {
                            // find the maximum cosine of the angle between joint edges
                            double cosine = Math.abs(angle(approx.toArray()[j%4], approx.toArray()[j-2], approx.toArray()[j-1]));
                            maxCosine = Math.max(maxCosine, cosine);
                        }

                        // if cosines of all angles are small
                        // (all angles are ~90 degree) then write quandrange
                        // vertices to resultant sequence
                        if( maxCosine < 0.3 )
                            squares.add(approx);
                    }
                }
            }
        }
    }

    // helper function:
    // finds a cosine of angle between vectors
    // from pt0->pt1 and from pt0->pt2
    double angle( Point pt1, Point pt2, Point pt0 ) {
        double dx1 = pt1.x - pt0.x;
        double dy1 = pt1.y - pt0.y;
        double dx2 = pt2.x - pt0.x;
        double dy2 = pt2.y - pt0.y;
        return (dx1*dx2 + dy1*dy2)/Math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
    }

    void extractChannel(Mat source, Mat out, int channelNum) {
        List<Mat> sourceChannels=new ArrayList<Mat>();
        List<Mat> outChannel=new ArrayList<Mat>();

        Core.split(source, sourceChannels);

        outChannel.add(new Mat(sourceChannels.get(0).size(),sourceChannels.get(0).type()));

        Core.mixChannels(sourceChannels, outChannel, new MatOfInt(channelNum,0));

        Core.merge(outChannel, out);
    }

    MatOfPoint approxPolyDP(MatOfPoint curve, double epsilon, boolean closed) {
        MatOfPoint2f tempMat=new MatOfPoint2f();

        Imgproc.approxPolyDP(new MatOfPoint2f(curve.toArray()), tempMat, epsilon, closed);

        return new MatOfPoint(tempMat.toArray());
    }
}
