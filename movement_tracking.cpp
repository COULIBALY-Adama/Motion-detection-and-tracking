////************************************************************************************************************////
////                          Motion detection and tracking	          		   				    	    ////
////                         																	      			////
////																											////
////             Author: COULIBALY Adama                             											////
////																											////
////             Compilation: 1- make      												 						////
////						  2- ./suivi_mouvement nom_video nb_sequences seuil_detection seuil_correspondance	////
////																											////
////		     Description: This program allows you to follow the movements of objects in a video  			////
////                                   																			////
////************************************************************************************************************////

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

// definition of a data structure for encompassed boxes
typedef struct {
	// Point 1 of the rectangle surrounding the object
	int x1;
	int y1;
	// Point 2 of the rectangle surrounding the object
	int x2;
	int y2;
	// Variable used to distinguish different object
	int indice;
} Enclosing_Box;

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
		cout << "Unable to play vide : " << video_name << endl;
		exit(0);
	} else {
		// Dimensions of vacuum imageso
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
				cout << "Fin de lecture de la video" << endl;
				exit(0);
			} else if (countFrame < nb_sequences) {

				// Converting Images to Grayscale
				cvtColor(frame, frameGray, CV_BGR2GRAY);

				// Saving the image to the vector
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
				vector<int> pixel_vector;

				// Retrieving pixel values in all images
				for (int k = 0; k < nb_images; k++) {
					int val = images_video[k].at<uchar>(i, j);
					pixel_vector.push_back(val);
				}

				// Sorting vector values
				std::sort(pixel_vector.begin(), pixel_vector.end());

				// Choice of median value
				background_image.at<uchar>(i, j) = pixel_vector[(nb_images
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
				cout << "Error while saving " << fileName
						<< endl;
		} else {
			cout << "Failed to extract the background" << endl;
		}

		return background_image;
	}
}

Mat detection_improvement(Mat &image_mouvement) {

	Mat image1 = image_mouvement.clone();

	Mat img_processed = image_mouvement.clone();
	vector<vector<Point> > contours;

	// Erosion + dilation

	Mat element_erosion = getStructuringElement(MORPH_RECT, Size(5, 5),
			Point(1, 1));
	Mat element_dilatation = getStructuringElement(MORPH_RECT, Size(3, 3),
			Point(1, 1));

	erode(img_processed, img_processed, element_erosion);
	dilate(img_processed, img_processed, element_dilatation);

	return image_mouvement;
}

// Function used to determine the boxes surrounding objects
vector<Enclosing_Box> determine_bounding_box(const Mat &frame_courant) {
	// List of objects in the current frame
	vector<Enclosing_Box> vecteur_boites_englobates;
	vecteur_boites_englobates.clear();

	vector<vector<Point> > contours;

	Mat frame = frame_courant.clone();

	// Determine the outlines of the different objects in the frame
	findContours(frame, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < (int) contours.size(); i++) {
		Mat pointMat(contours[i]);

		// We consider each contour as an object
		Rect rect = boundingRect(pointMat);

		Enclosing_Box enclosing_box;
		enclosing_box.x1 = rect.x;
		enclosing_box.y1 = rect.y;
		enclosing_box.x2 = rect.x + rect.width;
		enclosing_box.y2 = rect.y + rect.height;
		enclosing_box.indice = -1;
		vecteur_boites_englobates.push_back(enclosing_box);
	}
	return vecteur_boites_englobates;
}

// Function to draw a rectangle
void draw_bounding_box(Mat &frame_courant,
		vector<Enclosing_Box> vecteur_boites_englobates) {
	// Use green to draw boxes surrounding objects
	for (int i = 0; i < (int) vecteur_boites_englobates.size(); i++) {
		rectangle(frame_courant,
				Point(vecteur_boites_englobates[i].x1,
						vecteur_boites_englobates[i].y1),
				Point(vecteur_boites_englobates[i].x2,
						vecteur_boites_englobates[i].y2), CV_RGB(255, 0, 0), 1);
	}
}

// Function to draw circles on the image
void dessiner_cercle(Mat &image_suivi, Point centre, int d,
		const Scalar& color) {
	circle(image_suivi, centre, d, color, 1, 0);
}

// Function to draw squares on the image
void dessiner_carre(Mat &image_suivi, Point centre, int d,
		const Scalar& color) {
	rectangle(image_suivi, Point(centre.x - d, centre.y - d),
			Point(centre.x + d, centre.y + d), color, 1, 0);
}

// Function to draw crosses on the image
void dessiner_croix(Mat &image_suivi, Point centre, int d,
		const Scalar& color) {
	// The shape is x for each point
	line(image_suivi, Point(centre.x - d, centre.y - d),
			Point(centre.x + d, centre.y + d), color, 1, 0);
	line(image_suivi, Point(centre.x + d, centre.y - d),
			Point(centre.x - d, centre.y + d), color, 1, 0);
}

// Initialize the Kalman Filter
void initialiser_filtre_kalman(map<int, KalmanFilter> &liste_filtres_kalman,
		vector<Enclosing_Box> &vecteur_boites_englobates, int width,
		int height, string video_name) {
	int maxIndex = -1;

	// Determine the maximum index of the object in the list
	for (int i = 0; i < (int) vecteur_boites_englobates.size(); i++) {
		if (maxIndex < vecteur_boites_englobates[i].indice
				&& vecteur_boites_englobates[i].indice != -1) {
			maxIndex = vecteur_boites_englobates[i].indice;
		}
	}

	Mat mesure = Mat::zeros(4, 1, CV_32FC1);

	// Declare an image to contain all traces of all moving objects
	stringstream ss;
	ss << "images_suivi/" << video_name << ".png";
	string fileName = ss.str();
	Mat imgSuiviMouvement = imread(fileName, -1);

	for (int i = 0; i < (int) vecteur_boites_englobates.size(); i++) {

		// Initialize the Kalman filter
		KalmanFilter filtre_kalman(4, 4, 0);

		// Initialize matrices
		setIdentity(filtre_kalman.transitionMatrix, cvRealScalar(1));
		filtre_kalman.transitionMatrix.at<float>(0, 2) = 1;
		filtre_kalman.transitionMatrix.at<float>(1, 3) = 1;
		setIdentity(filtre_kalman.processNoiseCov, cvRealScalar(0));
		setIdentity(filtre_kalman.measurementNoiseCov, cvRealScalar(0));
		setIdentity(filtre_kalman.measurementMatrix, cvRealScalar(1));
		setIdentity(filtre_kalman.errorCovPost, cvRealScalar(1));

		// Make the prediction without relying on historical data
		Mat predictionMat = filtre_kalman.predict();

		// Measured
		mesure.at<float>(0, 0) = (vecteur_boites_englobates[i].x2
				+ vecteur_boites_englobates[i].x1) / 2;
		mesure.at<float>(1, 0) = (vecteur_boites_englobates[i].y2
				+ vecteur_boites_englobates[i].y1) / 2;
		// La vitesse vx, vy
		mesure.at<float>(2, 0) = 0;
		mesure.at<float>(3, 0) = 0;

		// Correction of measurement
		Mat correctionMat = filtre_kalman.correct(mesure);

		// For each object, initialize a tracked imag
		Mat imgSuivi = Mat::zeros(height, width, CV_8UC3);

		if (vecteur_boites_englobates[i].indice == -1) {
			// Reset object index
			maxIndex++;
			vecteur_boites_englobates[i].indice = maxIndex;

			// Draw the prediction path
			dessiner_croix(imgSuivi,
					Point(predictionMat.at<float>(0, 0),
							predictionMat.at<float>(1, 0)), 3,
					CV_RGB(0, 0, 255));
			dessiner_croix(imgSuiviMouvement,
					Point(predictionMat.at<float>(0, 0),
							predictionMat.at<float>(1, 0)), 3,
					CV_RGB(0, 0, 255));

			// Draw the measurement path
			dessiner_cercle(imgSuivi,
					Point(mesure.at<float>(0, 0), mesure.at<float>(1, 0)), 3,
					CV_RGB(0, 255, 0));
			dessiner_cercle(imgSuiviMouvement,
					Point(mesure.at<float>(0, 0), mesure.at<float>(1, 0)), 3,
					CV_RGB(0, 255, 0));

			// Draw the correction path
			dessiner_carre(imgSuivi,
					Point(correctionMat.at<float>(0, 0),
							correctionMat.at<float>(1, 0)), 3,
					CV_RGB(255, 0, 0));
			dessiner_carre(imgSuiviMouvement,
					Point(correctionMat.at<float>(0, 0),
							correctionMat.at<float>(1, 0)), 3,
					CV_RGB(255, 0, 0));

			// Draw texts in the motion tracking image
			stringstream ssText;
			ssText << vecteur_boites_englobates[i].indice;
			string text = ssText.str();
			int fontFace = CV_FONT_HERSHEY_SIMPLEX;
			double fontScale = 0.5;
			int thickness = 1;

			Point textPosition(mesure.at<float>(0, 0), mesure.at<float>(1, 0));
			putText(imgSuivi, text, textPosition, fontFace, fontScale,
					CV_RGB(255, 255, 255), thickness, 8);

			// Save the tracked image for this object
			stringstream ss;
			ss << "images_suivi/" << video_name << "_objet_"
					<< vecteur_boites_englobates[i].indice << ".png";
			string filename = ss.str();

			imwrite(filename, imgSuivi);

			// Add the current kalman filter to the filter list
			liste_filtres_kalman[vecteur_boites_englobates[i].indice] =
					filtre_kalman;
		}

	}
	imwrite(fileName, imgSuiviMouvement);
}

// Function to compare the objects of a previous frame and the current frame 
int chercherIndiceObjet(Enclosing_Box objet_precedent,
		vector<Enclosing_Box> &vecteur_objets_actuels,
		int seuil_correspondance) {
	// Initialize the index of the object to -1: i.e. not corresponding to any object 
	int indice = -1;

	float minDist = 1000000000.0;
	Enclosing_Box objet_considere;
	int centreX1 = (objet_precedent.x1 + objet_precedent.x2) / 2;
	int centreY1 = (objet_precedent.y1 + objet_precedent.y2) / 2;

	for (int i = 0; i < (int) vecteur_objets_actuels.size(); i++) {
		int centreX = (vecteur_objets_actuels[i].x1
				+ vecteur_objets_actuels[i].x2) / 2;
		int centreY = (vecteur_objets_actuels[i].y1
				+ vecteur_objets_actuels[i].y2) / 2;

		// check the distance between the objects: previous object and each object in the list
		float distance = sqrt(
				(centreX - centreX1) * (centreX - centreX1)
						+ (centreY - centreY1) * (centreY - centreY1));
		if (distance < minDist && distance < (float) seuil_correspondance) {
			minDist = distance;
			indice = i;
			objet_considere = vecteur_objets_actuels[i];
			// listObjetsActuels[i].indice=objetPrecedent.indice;
		}
		// In case of a tie check the smallest surface

	}
	return indice;
}

void detection_suivi_mouvement(string video_name, Mat background_image,
		int seuil_detection1, int seuil_correspondance) {

	vector<Mat> images_video;
	vector<Mat> images_mouvement;
	stringstream video_path;
	Mat imageSuivi;
	//imageMouv  = Mat::zeros(frame.size(), CV_8UC1);

	video_path << "videos/" << video_name;
	string fileName = video_path.str();
	char key;

	// Vector containing the detected objects of the previous frame
	vector<Enclosing_Box> ListObjetsPrecedents;
	ListObjetsPrecedents.clear();

	//Vector containing the objects detected in the current frame 
	vector<Enclosing_Box> ListObjetsActuel;
	ListObjetsActuel.clear();

	// Vector containing all objects
	vector<Enclosing_Box> ListObjetsTotal;
	ListObjetsTotal.clear();

	// List of Kalmam filter moving objects in the current frame
	map<int, KalmanFilter> listKalmanFilter;
	listKalmanFilter.clear();

	namedWindow(video_name, CV_WINDOW_AUTOSIZE);
	namedWindow("Image Arriere Plan", CV_WINDOW_AUTOSIZE);
	namedWindow("Detection Mouvement", CV_WINDOW_AUTOSIZE);

	GaussianBlur(background_image, background_image, Size(5, 5), 0, 0);
	// Loading video
	VideoCapture videoCapture(fileName);

	if (!videoCapture.isOpened()) {
		cout << "Unable to play vide : " << video_name << endl;
		exit(0);
	} else {
		// reading of the images constituting the video
		int numFrameActuel = 0;
		while (key != 'q' && key != 'Q') {

			Mat frame, frame_gray;

			videoCapture >> frame;

			if (!frame.data) {
				cout << "Fin de lecture de la video" << endl;
				break;
			} else {

				// Converting Images to Grayscale
				cvtColor(frame, frame_gray, CV_BGR2GRAY);

				// Saving the Image to the Image Vector
				images_video.push_back(frame_gray);

				GaussianBlur(images_video.back(), images_video.back(),
						Size(5, 5), 0, 0);
				Mat difference_images;
				// Difference between background image and captured image to detect motion
				absdiff(background_image, images_video.back(),
						difference_images);
				images_mouvement.push_back(difference_images);
				// GaussianBlur( images_mouvement.back(), images_mouvement.back(),Size( 3, 3 ), 0, 0 );
				// Binary threshold to eliminate noise
				threshold(images_mouvement.back(), images_mouvement.back(),
						seuil_detection1, 255.0, CV_THRESH_BINARY);
				images_mouvement.back() = detection_improvement(
						images_mouvement.back());

				// determination of bounding boxes for detected objects
								vector<Enclosing_Box> vecteur_boites_englobantes =
										determine_bounding_box(images_mouvement.back());
								draw_bounding_box(frame, vecteur_boites_englobantes);

				if (images_mouvement.back().data) {
					// Recording the motion image
					stringstream path;
					path << "images_mouvement/" << video_name << "seuil_"
							<< seuil_detection1 << "_frame_"
							<< images_video.size() - 1 << ".png";
					string fileName = path.str();


				// Determine objects located in the current frame
				ListObjetsActuel = determine_bounding_box(
						images_mouvement.back());

				// Draw boxes on the corresponding image
				draw_bounding_box(frame, ListObjetsActuel);

				imageSuivi = Mat::zeros(frame.size(), CV_8UC3);

				//*****************************************  Motion tracking   *************************************/
				if (numFrameActuel == 0 && (ListObjetsActuel.size() > 0)) {
					initialiser_filtre_kalman(listKalmanFilter,
							ListObjetsActuel, frame.cols, frame.rows,
							video_name);
					ListObjetsPrecedents = ListObjetsActuel;
					ListObjetsTotal = ListObjetsActuel;
				} else

				{
					stringstream ss;
					ss << "images_suivi/" << video_name << ".png";
					string fileName = ss.str();

					imageSuivi = imread(fileName, -1); // loading the image to trace the route
					ListObjetsTotal.clear();

					for (int i = 0; i < (int) ListObjetsPrecedents.size();
							i++) {
						// 1st Step: Predict the positions of the objects
						Mat predictionMat =
								listKalmanFilter[ListObjetsPrecedents[i].indice].predict();

						stringstream ss;
						ss << "images_suivi/" << video_name << "_objet_"
								<< ListObjetsPrecedents[i].indice << ".png";
						string fileName = ss.str();
						Mat imgSuiviObj = imread(fileName, -1);

						// Draw the prediction path
						dessiner_croix(imgSuiviObj,
								Point(predictionMat.at<float>(0, 0),
										predictionMat.at<float>(1, 0)), 3,
								CV_RGB(255, 0, 0));
						dessiner_croix(imageSuivi,
								Point(predictionMat.at<float>(0, 0),
										predictionMat.at<float>(1, 0)), 3,
								CV_RGB(255, 0, 0));

						// Find the correspondence between the objects of the previous frame and those of the current fram
						int correspondance = chercherIndiceObjet(
								ListObjetsPrecedents[i], ListObjetsActuel,
								seuil_correspondance);

						if (correspondance != -1) {
							ListObjetsActuel[correspondance].indice =
									ListObjetsPrecedents[i].indice;

							// Step 2: Measure the positions of the objects
							Mat mesure = Mat::zeros(4, 1, CV_32FC1);

							mesure.at<float>(0, 0) =
									(ListObjetsActuel[correspondance].x1
											+ ListObjetsActuel[correspondance].x2)
											/ 2;
							mesure.at<float>(1, 0) =
									(ListObjetsActuel[correspondance].y1
											+ ListObjetsActuel[correspondance].y2)
											/ 2;

							float vx = 0;
							float vy = 0;

							// Speed coordinates
							vx = ((ListObjetsActuel[correspondance].x1
									+ ListObjetsActuel[correspondance].x2) / 2)
									- ((ListObjetsPrecedents[i].x1
											+ ListObjetsPrecedents[i].x2) / 2);

							vy = ((ListObjetsActuel[correspondance].y1
									+ ListObjetsActuel[correspondance].y2) / 2)
									- ((ListObjetsPrecedents[i].y1
											+ ListObjetsPrecedents[i].y2) / 2);

							mesure.at<float>(2, 0) = vx;
							mesure.at<float>(3, 0) = vy;

							// Display and save the results, follow the movement and draw the measurement path
							dessiner_cercle(imageSuivi,
									Point(mesure.at<float>(0, 0),
											mesure.at<float>(1, 0)), 3,
									CV_RGB(0, 255, 0));
							dessiner_cercle(imgSuiviObj,
									Point(mesure.at<float>(0, 0),
											mesure.at<float>(1, 0)), 3,
									CV_RGB(0, 255, 0));

							// Step 3: Correct the positions of the objects
							Mat correctionMat =
									listKalmanFilter[ListObjetsPrecedents[i].indice].correct(
											mesure);

							// Draw the path of the correction
							dessiner_carre(imageSuivi,
									Point(correctionMat.at<float>(0, 0),
											correctionMat.at<float>(1, 0)), 3,
									CV_RGB(0, 0, 255));
							dessiner_carre(imgSuiviObj,
									Point(correctionMat.at<float>(0, 0),
											correctionMat.at<float>(1, 0)), 3,
									CV_RGB(0, 0, 255));

							// Save the tracked image for this object
							imwrite(fileName, imgSuiviObj);
							ListObjetsPrecedents[i] =
									ListObjetsActuel[correspondance];
						}
					}

					ListObjetsTotal = ListObjetsPrecedents;

					for (int i = 0; i < (int) ListObjetsActuel.size(); i++) {
						if (ListObjetsActuel[i].indice == -1) {
							int correspondance = chercherIndiceObjet(
									ListObjetsActuel[i], ListObjetsPrecedents,
									seuil_correspondance);

							if (correspondance != -1) {
								ListObjetsActuel[i].indice =
										ListObjetsPrecedents[correspondance].indice;
							}
						}
					}

					for (int i = 0; i < (int) ListObjetsActuel.size(); i++) {
						if (ListObjetsActuel[i].indice == -1)
							ListObjetsTotal.push_back(ListObjetsActuel[i]);
					}

					initialiser_filtre_kalman(listKalmanFilter, ListObjetsTotal,
							frame.cols, frame.rows, video_name);
					ListObjetsPrecedents = ListObjetsTotal;
				}

				// Add the names of moving objects in the current frame 
				for (int i = 0; i < (int) ListObjetsActuel.size(); i++) {
					stringstream ssTmp;
					ssTmp << ListObjetsActuel[i].indice;
					string text = ssTmp.str();

					int fontFace = CV_FONT_HERSHEY_SIMPLEX;
					double fontScale = 0.5;
					int thickness = 1;

					int x = (ListObjetsActuel[i].x1 + ListObjetsActuel[i].x2)
							/ 2;
					int y = (ListObjetsActuel[i].y1 + ListObjetsActuel[i].y2)
							/ 2;
					Point textPosition(x, y);
					putText(frame, text, textPosition, fontFace, fontScale,
							CV_RGB(0, 0, 255), thickness, 8);
				}

				// Viewing and Saving Results 
				stringstream ss1;
				ss1 << "images_suivi/" << video_name << ".png";
				string fileName1 = ss1.str();
				if (!imwrite(fileName1, imageSuivi))
					cout << "Error while saving " << fileName1
							<< endl;

						stringstream ss2;
				ss2 << "images_videos/" << video_name << "_" << numFrameActuel
						<< ".png";
				string fileName2 = ss2.str();
				if (!imwrite(fileName2, frame))
					cout << "Error while saving " << fileName2
							<< endl;

				// recording of the video image with bounding boxes
								stringstream ss3;
								ss3 << "images_mouvement/" << video_name << "seuil_" << seuil_detection1
										<< "_frame_" << images_video.size() - 1 << ".png";
								string fileName3 = ss3.str();
								if (!imwrite(fileName3, images_mouvement.back()))
									cout << "Error while saving " << fileName3
											<< endl;

				imshow("Suivi Mouvement", imageSuivi);

				numFrameActuel++;

				imshow(video_name,frame);
				imshow("Image Arriere Plan", background_image);
				imshow("Detection Mouvement", images_mouvement.back());

				key = cvWaitKey(40);

			}
		}

	}
	cvDestroyAllWindows();

}
}

//main function
int main(int argc, char** argv)
{

	// Retrieving program parameters
	string video_name = argv[1];  // Video name
	int nb_sequences = atoi(argv[2]); // Number of image sequences
	int seuil_detection = atoi(argv[3]); // image detection threshold
	int seuil_correspondance = atoi(argv[4]); // match threshold

	Mat background_image;

	background_image = background_extraction(video_name, nb_sequences);

	detection_suivi_mouvement(video_name, background_image, seuil_detection,
			seuil_correspondance);

	waitKey(0);

	return 0;
}
