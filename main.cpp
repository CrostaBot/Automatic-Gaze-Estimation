#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
 
#include <iostream>
#include <stdio.h>
#include <map>
 
using namespace std;
using namespace cv;

void drawHist(const vector<int>& data, Mat3b& dst, int binSize = 100, int height = 0)
{
    int max_value = *max_element(data.begin(), data.end());
    int rows = 0;
    int cols = 0;
    if (height == 0) {
        rows = max_value + 10;
    } else { 
        rows = max(max_value + 10, height);
    }

    cols = data.size() * binSize;

    dst = Mat3b(rows, cols, Vec3b(0,0,0));

    for (int i = 0; i < data.size(); ++i)
    {
        int h = rows - data[i];
        rectangle(dst, Point(i*binSize, h), Point((i + 1)*binSize-1, rows), (i%2) ? Scalar(0, 100, 255) : Scalar(0, 0, 255), FILLED);
    }

}

int main( )
{
    Mat image;
    image = imread("/Users/Nicola/Library/Mobile Documents/com~apple~CloudDocs/Desktop/ProjectCV/images_T2S/11.jpg", IMREAD_COLOR);  

    //Applying gaussian filter to smooth and facility face and eye detection
    Mat image_blur, image_gray;
    cvtColor(image, image_gray, COLOR_BGR2GRAY);
    GaussianBlur(image, image_blur, Size(3, 3), 0);
 
    // Load face and eyes cascade (.xml file)
    CascadeClassifier face_cascade;
    CascadeClassifier eyes_cascade;
    face_cascade.load( "/Users/Nicola/Library/Mobile Documents/com~apple~CloudDocs/Desktop/ProjectCV/xml/haarcascade_frontalface_alt2.xml" );
    eyes_cascade.load( "/Users/Nicola/Library/Mobile Documents/com~apple~CloudDocs/Desktop/ProjectCV/xml/haarcascade_eye.xml" );

    // Detect faces
    vector<Rect> faces;
    map<int, Mat> crop_eyes;
    face_cascade.detectMultiScale(image_blur, faces, 1.01, 3, 0| CASCADE_SCALE_IMAGE, Size(200, 200));
 
    //In each face, detect eyes and draw rectangle
    Mat roi, roi_gray;
    for(size_t i = 0; i < faces.size(); i++)
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        //ellipse( image_blur, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 0, 0, 0 ), 2, 8, 0 );
        Mat face_gray = image_gray(faces[i]);
        
        vector<Rect> eyes;
        Rect crop_eye;

        eyes_cascade.detectMultiScale(face_gray, eyes, 1.01, 3, 0 | CASCADE_SCALE_IMAGE, Size(50, 50));
        for( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center(faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2);
            int radius = cvRound((eyes[j].width + eyes[j].height)*0.07);
            double height = eyes[j].height;
            double width = eyes[j].width;
            Point eye_vertex_1(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y);
            Point eye_vertex_2(faces[i].x + eyes[j].x + width, faces[i].y + eyes[j].y + height);

            // Save only one eye in cropping region of interest
            if (j < 2) 
            {   
                crop_eye = Rect(eye_vertex_1.x + 2, eye_vertex_1.y + 2, height - 3, width - 3);
            }

            //rectangle(image_blur, eye_vertex_1, eye_vertex_2, Scalar( 0, 0, 0 ), 2);
            roi = Mat(image_blur, crop_eye);
            cvtColor( roi, roi_gray, COLOR_BGR2GRAY );
            crop_eyes[i] = roi_gray;
        }
    }    

    // Apply threshold to highlights dark region
    Mat roi_thres;		
    threshold(roi_gray, roi_thres, 60, 255, 0);
    //imshow("threshold image", roi_thres);

    //Find edges with Canny to undestand eye contour
    Mat canny_output;
    Canny(roi_thres, canny_output, 50, 200);
    vector<vector<Point> > contours;
    findContours(canny_output, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));

    vector<RotatedRect> minEllipse(contours.size());
    for(size_t i = 0; i < contours.size(); i++)
    {
        if(contours[i].size() > 5)
        {
            minEllipse[i] = fitEllipse(contours[i]);
        }
    }

   	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
   	double a = minEllipse[0].size.width;
   	RotatedRect eye_contour;
    for(size_t i = 0; i<contours.size(); i++)
    {
        // Select biggest ellipse
        if (minEllipse[i+1].size.width > a)
       	{
       		eye_contour = minEllipse[i+1];
       		a = minEllipse[i+1].size.width;
       	}
    }

    //ellipse(roi_thres, eye_contour, Scalar(0, 0, 0), 2);
    //imshow("Canny detection", roi_thres);

    // Extract new roi
 	Mat M, roi_rotated, roi_cropped;
   	float angle = eye_contour.angle;
    Size rect_size = eye_contour.size;
    if (eye_contour.angle < -45.) {
        angle += 90.0;
        swap(rect_size.width, rect_size.height);
    }
    M = getRotationMatrix2D(eye_contour.center, angle, 1.0);
    warpAffine(roi_thres, roi_rotated, M, roi_thres.size(), INTER_CUBIC);
    getRectSubPix(roi_rotated, rect_size, eye_contour.center, roi_cropped);

    rotate(roi_cropped, roi_cropped, cv::ROTATE_90_CLOCKWISE);

    //imshow("Crop and rotate", roi_cropped);

    // Average of all intensity value gray scale of the eye
    vector<int> averages;
    Scalar intensity;
    for( size_t x = 0; x < roi_cropped.size().width; x++)
    {
        int intensities = 0;
        for( size_t y = 0; y < roi_cropped.size().height; y++)
        {
            intensity = roi_cropped.at<uchar>(y, x);
            int gray_value = intensity.val[0];
            intensities = intensities + gray_value;
        }

        int average = intensities/roi_cropped.size().height;
        averages.push_back(average);
    }

    vector<int> bin_right, bin_center, bin_left;
    int size_bin;
	size_bin = (int) averages.size()/3;
    for( size_t x = 0; x < averages.size(); x++)
    {
    	if (x < size_bin)
    		bin_right.push_back(averages[x]);
    	else if (x >= size_bin && x < size_bin*2) 
    		bin_center.push_back(averages[x]);
    	else
    		bin_left.push_back(averages[x]);
    }

    vector<int> hist_values;
    int mean1 = 0;
    int mean_right, mean_center, mean_left;
    for( size_t x = 0; x < bin_right.size(); x++)
    {
    	mean1 = mean1 + bin_right[x];
    }
    mean_right =(int) mean1/bin_right.size();
    hist_values.push_back(mean_right);

    int mean2 = 0;
    for( size_t x = 0; x < bin_center.size(); x++)
    {
    	mean2 = mean2 + bin_center[x];
    }
    mean_center = (int) mean2 / bin_center.size();
    hist_values.push_back(mean_center);

    int mean3 = 0;
    
    for( size_t x = 0; x < bin_left.size(); x++)
    {
    	mean3 = mean3 + bin_left[x];
    }
    mean_left = (int) mean3 / bin_left.size();
    hist_values.push_back(mean_left);

    if (mean_left < mean_right  && mean_left < mean_center)
    	cout << "Looking left" << endl;
     if (mean_center < mean_right  && mean_center < mean_left)
    	cout << "Looking straight" << endl;
     if (mean_right < mean_left  && mean_right < mean_center)
    	cout << "Looking right" << endl;

    // Create histogram
    Mat3b hist_tot, hist;
    drawHist(averages, hist_tot);
    drawHist(hist_values, hist);

    // Show image
    imshow("Detected Face and Eyes", image_blur);
    imshow("Eye region", roi_cropped);
    imshow("Canny", roi_thres);
    imshow("Histogram completo", hist_tot);
    imshow("Histogram", hist);
   
    waitKey(0);                   
    return 0;
}