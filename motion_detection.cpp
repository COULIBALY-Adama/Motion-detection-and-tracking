////****************************************************************************************************////
////                          Motion detection and tracking		      	    		   				    ////
////                         											 						        ////
////																									////
////             Author: COULIBALY Adama                             									////
////																									////
////             Compilation: 1- make      												 				////
////						  2- ./motion_detection video_name nb_sessions detection_threshold		    ////
////																									////
////		     Description: This program can detect the movements of objects in a video. 			    ////
////                                   																	////
////****************************************************************************************************////

#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <bits/basic_string.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

using namespace cv;
using namespace std;

//definition of a data structure for bounding boxes
//supposed to surround objects
typedef struct {
	//Point 1 of the rectangle surrounding the object
	int x1;
	int y1;
	//Point 2 of the rectangle surrounding the object
	int x2;
	int y2;

} Enclosing_Box;

// Function used to determine the boxes surrounding objects
vector<Enclosing_Box> determine_enclosing_box(const Mat &current_frame) {
	// List of objects in the current frame
	vector<Enclosing_Box> vector_boxes_englobates;
	vector_boxes_englobates.clear();

	vector<vector<Point> > outlines;

	Mat frame = current_frame.clone();

	// Determine the outlines of the different objects in the frame
	findContours(frame, outlines, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < (int) outlines.size(); i++) {
		Mat pointMat(outlines[i]);

		// We consider each contour as an object
		Rect rect = boundingRect(pointMat);

		Enclosing_Box enclosing_box;
		enclosing_box.x1 = rect.x;
		enclosing_box.y1 = rect.y;
		enclosing_box.x2 = rect.x + rect.width;
		enclosing_box.y2 = rect.y + rect.height;
		vector_boxes_englobates.push_back(enclosing_box);
	}
	return vector_boxes_englobates;
}

// Function to draw a rectangle
void draw_including_box(Mat &current_frame,
		vector<Enclosing_Box> vector_boxes_englobates) {
	// Use green to draw boxes surrounding objects
	for (int i = 0; i < (int) vector_boxes_englobates.size(); i++) {
		rectangle(current_frame,
				Point(vector_boxes_englobates[i].x1,
						vector_boxes_englobates[i].y1),
				Point(vector_boxes_englobates[i].x2,
						vector_boxes_englobates[i].y2), CV_RGB(255, 0, 0), 1);
	}
}

// Function for constructing the background image
Mat background_extraction(string video_name, int nb_sequences) {
	int height, width;
	stringstream video_path;
	video_path << "videos/" << video_name;
	string fileName = video_path.str();
	Mat background_image;

	// Loading video
	VideoCapture videoCapture(fileName);

	if (!videoCapture.isOpened()) {
		cout << "Unable to play video : " << video_name << endl;
		exit(0);
	} else {
		// Video frame dimensions
		height = videoCapture.get(CV_CAP_PROP_FRAME_HEIGHT);
		width = videoCapture.get(CV_CAP_PROP_FRAME_WIDTH);

		// Initializing the background image
		background_image = Mat::zeros(height, width, CV_8UC1);

		// Vector containing the sequence of images for the construction of the background
		vector<Mat> images_video;
		int countFrame = 0;

		while (true) {
			Mat frame, frameGray;

			videoCapture >> frame;

			if (!frame.data) {
				cout << "End of video playback" << endl;
				exit(0);
			} else if (countFrame < nb_sequences) {

				// Converting Images to Grayscale
				cvtColor(frame, frameGray, CV_BGR2GRAY);

				// Saving the Image to the Image Vector
				images_video.push_back(frameGray);

			} else if (countFrame >= nb_sequences)
				break;

			countFrame++;
		}

		int nb_images = images_video.size();

		// Creating the background
		for (int i = 0; i < background_image.rows; i++) {
			for (int j = 0; j < background_image.cols; j++) {
				// Vector containing the values in all the images of a given pixel
				vector<int> vecteur_pixel;

				// Retrieving pixel values in all images
				for (int k = 0; k < nb_images; k++) {
					int val = images_video[k].at<uchar>(i, j);
					vecteur_pixel.push_back(val);
				}

				// Sorting vector values
				std::sort(vecteur_pixel.begin(), vecteur_pixel.end());

				// Choice of median value
				background_image.at<uchar>(i, j) = vecteur_pixel[(nb_images
						+ 1) / 2];
			}
		}
		if (background_image.data) {
			// Background recording
			stringstream path;
			path << "background/" << video_name << "_" << nb_sequences
					<< ".png";
			string fileName = path.str();
			if (!imwrite(fileName, background_image))
				cout << "Error while registering " << fileName
						<< endl;
		} else {
			cout << "Failed to extract the background" << endl;
		}

		return background_image;
	}
}

// Function to enhance the image corresponding to the detected motion
Mat improvement_dection(Mat &motion_image) {

	// Erosion + dilation

	Mat element_erosion = getStructuringElement(MORPH_RECT, Size(3, 3),
			Point(1, 1));
	Mat dilation_element = getStructuringElement(MORPH_RECT, Size(3, 3),
			Point(1, 1));

	erode(motion_image, motion_image, element_erosion);
	dilate(motion_image, motion_image, dilation_element);

	return motion_image;
}

// Function to detect movements
vector<Mat> detection_mouvement(string video_name, Mat background_image,
		int seuil) {

	// definition of variables
	vector<Mat> images_video;
	vector<Mat> images_mouvement;

	stringstream video_path;
	video_path << "videos/" << video_name;
	string fileName = video_path.str();

	char key;
	namedWindow(video_name, CV_WINDOW_AUTOSIZE);
	namedWindow("Background Image", CV_WINDOW_AUTOSIZE);
	namedWindow("Motion detection", CV_WINDOW_AUTOSIZE);

	// application of the Gaussian mask for smoothing
	GaussianBlur(background_image, background_image, Size(5, 5), 0, 0);

	// Loading video
	VideoCapture videoCapture(fileName);

	// load check
	if (!videoCapture.isOpened()) {
		cout << "Unable to play video : " << video_name << endl;
		exit(0);
	} else {
		// playback of the images constituting the video

		while (key != 'q' && key != 'Q') {

			Mat frame, frame_gray;

			videoCapture >> frame;

			// end of video test
			if (!frame.data) {
				cout << "End of video playback" << endl;
				break;
			} else {

				// Converting Images to Grayscale
				cvtColor(frame, frame_gray, CV_BGR2GRAY);

				// Saving the Image to the Image Vector
				images_video.push_back(frame_gray);

				// application of the Gaussian mask for smoothing
				GaussianBlur(images_video.back(), images_video.back(),
						Size(5, 5), 0, 0);
				Mat difference_images;
				// Difference between background image and captured image to detect motion
				absdiff(background_image, images_video.back(),
						difference_images);
				images_mouvement.push_back(difference_images);

				// Binary threshold to eliminate noise
				threshold(images_mouvement.back(), images_mouvement.back(),
						seuil, 255.0, CV_THRESH_BINARY);
				images_mouvement.back() = improvement_dection(
						images_mouvement.back());

				// determination of bounding boxes for detected objects
				vector<Enclosing_Box> bounding_boxes_vector =
						determine_enclosing_box(images_mouvement.back());
				draw_including_box(frame, bounding_boxes_vector);

				// recording of results
				if (images_mouvement.back().data) {
					// Recording the motion image
					stringstream path;
					path << "images_mouvement/" << video_name << "seuil_"
							<< seuil << "_frame_" << images_video.size() - 1
							<< ".png";
					string fileName = path.str();
					if (!imwrite(fileName, images_mouvement.back()))
						cout << "Error while registering "
								<< fileName << endl;
				} else {
					cout << "Failed to determine motion for frame "
							<< images_video.size() - 1 << endl;
				}

				// recording of the video image with bounding boxes
				stringstream path1;
				path1 << "images_videos/" << video_name << "seuil_" << seuil
						<< "_frame_" << images_video.size() - 1 << ".png";
				string fileName1 = path1.str();
				if (!imwrite(fileName1, frame))
					cout << "Error while registering " << fileName1
							<< endl;

				// displays of the different images
				imshow(video_name, frame);
				imshow("Background Image", background_image);
				imshow("Motion detection", images_mouvement.back());

				key = cvWaitKey(40);

			}
		}

	}
	cvDestroyAllWindows();
	return images_mouvement;

}

// main function
int main(int argc, char** argv) {

	// Retrieving program parameters
	string video_name = argv[1];  // Video name
	int nb_sequences = atoi(argv[2]); // Number of image sequences
	int seuil = atoi(argv[3]); // Detection limit

	Mat background_image;

	background_image = background_extraction(video_name, nb_sequences);

	vector<Mat> images_mouvement;

	images_mouvement = detection_mouvement(video_name, background_image,
			seuil);

	waitKey(0);

	return 0;
}
