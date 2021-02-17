////****************************************************************************************************////
////                          Détection et suivi de mouvement	      	    		   				    ////
////                         											 						        ////
////																									////
////             Author: COULIBALY Adama                             									////
////																									////
////             Compilation: 1- make      												 				////
////						  2- ./detection_mouvement nom_video nb_seuances seuil_detection		    ////
////																									////
////		     Description: Ce programme permet de détecter les mouvements des objets dans une vidéo  ////
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

//définition d'unstructure de donnée pour les boites englobantes
//censées entourer les objets
typedef struct {
	//Point 1 du rectangle entourant l'objet
	int x1;
	int y1;
	//Point 2 du rectangle entourant l'objet
	int x2;
	int y2;

} Boite_Englobante;

// Fonction permettant de determiner les boites encadrant les objets
vector<Boite_Englobante> determiner_boite_englobante(const Mat &frame_courant) {
	// Liste des objets se trouvant dans le frame actuel
	vector<Boite_Englobante> vecteur_boites_englobates;
	vecteur_boites_englobates.clear();

	vector<vector<Point> > contours;

	Mat frame = frame_courant.clone();

	// Déterminer les contours des différents objets du frame
	findContours(frame, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < (int) contours.size(); i++) {
		Mat pointMat(contours[i]);

		//On considere chaque contour comme un objet
		Rect rect = boundingRect(pointMat);

		Boite_Englobante boite_englobante;
		boite_englobante.x1 = rect.x;
		boite_englobante.y1 = rect.y;
		boite_englobante.x2 = rect.x + rect.width;
		boite_englobante.y2 = rect.y + rect.height;
		vecteur_boites_englobates.push_back(boite_englobante);
	}
	return vecteur_boites_englobates;
}

// Fonction pour dessiner un rectangle
void dessiner_boite_englobante(Mat &frame_courant,
		vector<Boite_Englobante> vecteur_boites_englobates) {
	//Utiliser le vert pour dessiner des boites entourant des objets
	for (int i = 0; i < (int) vecteur_boites_englobates.size(); i++) {
		rectangle(frame_courant,
				Point(vecteur_boites_englobates[i].x1,
						vecteur_boites_englobates[i].y1),
				Point(vecteur_boites_englobates[i].x2,
						vecteur_boites_englobates[i].y2), CV_RGB(255, 0, 0), 1);
	}
}

// Fonction pour la construction de l'image d'arrière-plan
Mat extraction_arriere_plan(string nom_video, int nb_sequences) {
	int hauteur, largeur;
	stringstream chemin_video;
	chemin_video << "videos/" << nom_video;
	string fileName = chemin_video.str();
	Mat image_arriere_plan;

	// Chargement de la video
	VideoCapture videoCapture(fileName);

	if (!videoCapture.isOpened()) {
		cout << "Impossible de lire la video : " << nom_video << endl;
		exit(0);
	} else {
		// Dimensions des images de la video
		hauteur = videoCapture.get(CV_CAP_PROP_FRAME_HEIGHT);
		largeur = videoCapture.get(CV_CAP_PROP_FRAME_WIDTH);

		// Initialisation de l'image d'arrière-plan
		image_arriere_plan = Mat::zeros(hauteur, largeur, CV_8UC1);

		// Vecteur contenant les séquence d'images pour la construction de l'arrière-plan
		vector<Mat> images_video;
		int countFrame = 0;

		while (true) {
			Mat frame, frameGray;

			videoCapture >> frame;

			if (!frame.data) {
				cout << "Fin de lecture de la video" << endl;
				exit(0);
			} else if (countFrame < nb_sequences) {

				// Conversion des images en niveau de gris
				cvtColor(frame, frameGray, CV_BGR2GRAY);

				// Enregistrement de l'image dans le vecteur d'images
				images_video.push_back(frameGray);

			} else if (countFrame >= nb_sequences)
				break;

			countFrame++;
		}

		int nb_images = images_video.size();

		// Création de l'arrière-plan
		for (int i = 0; i < image_arriere_plan.rows; i++) {
			for (int j = 0; j < image_arriere_plan.cols; j++) {
				// Vecteur contenant les valeurs dans toutes les images d'un pixel donné
				vector<int> vecteur_pixel;

				// Réccupération des valeurs du pixel dans toutes les images
				for (int k = 0; k < nb_images; k++) {
					int val = images_video[k].at<uchar>(i, j);
					vecteur_pixel.push_back(val);
				}

				// Tri des valeurs du vecteur
				std::sort(vecteur_pixel.begin(), vecteur_pixel.end());

				// Choix de la valeur mediane
				image_arriere_plan.at<uchar>(i, j) = vecteur_pixel[(nb_images
						+ 1) / 2];
			}
		}
		if (image_arriere_plan.data) {
			// Enregistrement de l'arrière plan
			stringstream path;
			path << "arriere_plan/" << nom_video << "_" << nb_sequences
					<< ".png";
			string fileName = path.str();
			if (!imwrite(fileName, image_arriere_plan))
				cout << "Erreur lors de l'enregistrement de " << fileName
						<< endl;
		} else {
			cout << "Echec de l'extraction de l'arriere plan" << endl;
		}

		return image_arriere_plan;
	}
}

//Fonction pour améliorer l'image correspondant au mouvement détecté
Mat amelioration_dection(Mat &image_mouvement) {

	// Erosion + dilatation

	Mat element_erosion = getStructuringElement(MORPH_RECT, Size(3, 3),
			Point(1, 1));
	Mat element_dilatation = getStructuringElement(MORPH_RECT, Size(3, 3),
			Point(1, 1));

	erode(image_mouvement, image_mouvement, element_erosion);
	dilate(image_mouvement, image_mouvement, element_dilatation);

	return image_mouvement;
}

//Fonction pour détecter les mouvements
vector<Mat> detection_mouvement(string nom_video, Mat image_arriere_plan,
		int seuil) {

	//définition des variables
	vector<Mat> images_video;
	vector<Mat> images_mouvement;

	stringstream chemin_video;
	chemin_video << "videos/" << nom_video;
	string fileName = chemin_video.str();

	char key;
	namedWindow(nom_video, CV_WINDOW_AUTOSIZE);
	namedWindow("Image Arriere Plan", CV_WINDOW_AUTOSIZE);
	namedWindow("Detection Mouvement", CV_WINDOW_AUTOSIZE);

	//application du masque gaussien pour le lissage
	GaussianBlur(image_arriere_plan, image_arriere_plan, Size(5, 5), 0, 0);

	// Chargement de la video
	VideoCapture videoCapture(fileName);

	//vérification du chargement
	if (!videoCapture.isOpened()) {
		cout << "Impossible de lire la video : " << nom_video << endl;
		exit(0);
	} else {
		// lecture des images constituant la video

		while (key != 'q' && key != 'Q') {

			Mat frame, frame_gray;

			videoCapture >> frame;

			//test de fin de video
			if (!frame.data) {
				cout << "Fin de lecture de la video" << endl;
				break;
			} else {

				// Conversion des images en niveau de gris
				cvtColor(frame, frame_gray, CV_BGR2GRAY);

				// Enregistrement de l'image dans le vecteur d'image
				images_video.push_back(frame_gray);

				//application du masque gaussien pour le lissage
				GaussianBlur(images_video.back(), images_video.back(),
						Size(5, 5), 0, 0);
				Mat difference_images;
				// Différence entre image arriere-plan et image capturée pour détecter le mouvement
				absdiff(image_arriere_plan, images_video.back(),
						difference_images);
				images_mouvement.push_back(difference_images);

				// Seuillage binaire pour éliminer les bruits
				threshold(images_mouvement.back(), images_mouvement.back(),
						seuil, 255.0, CV_THRESH_BINARY);
				images_mouvement.back() = amelioration_dection(
						images_mouvement.back());

				//détermination des boites englobantes pour les objets détectés
				vector<Boite_Englobante> vecteur_boites_englobantes =
						determiner_boite_englobante(images_mouvement.back());
				dessiner_boite_englobante(frame, vecteur_boites_englobantes);

				//enrégistrement des résultats
				if (images_mouvement.back().data) {
					// Enregistrement de l'image de mouvement
					stringstream path;
					path << "images_mouvement/" << nom_video << "seuil_"
							<< seuil << "_frame_" << images_video.size() - 1
							<< ".png";
					string fileName = path.str();
					if (!imwrite(fileName, images_mouvement.back()))
						cout << "Erreur lors de l'enregistrement de "
								<< fileName << endl;
				} else {
					cout << "Echec de détermination du mouvement pour le frame "
							<< images_video.size() - 1 << endl;
				}

				//enrégistrement de l'image de la vidéo avec les boites englobantes
				stringstream path1;
				path1 << "images_videos/" << nom_video << "seuil_" << seuil
						<< "_frame_" << images_video.size() - 1 << ".png";
				string fileName1 = path1.str();
				if (!imwrite(fileName1, frame))
					cout << "Erreur lors de l'enregistrement de " << fileName1
							<< endl;

				//affichages des différentes images
				imshow(nom_video, frame);
				imshow("Image Arriere Plan", image_arriere_plan);
				imshow("Detection Mouvement", images_mouvement.back());

				key = cvWaitKey(40);

			}
		}

	}
	cvDestroyAllWindows();
	return images_mouvement;

}

//fonction principale
int main(int argc, char** argv) {

	// Réccupération des paramètres du programme
	string nom_video = argv[1];  //Nom de la video
	int nb_sequences = atoi(argv[2]); //Nombre de séquence d'images
	int seuil = atoi(argv[3]); //Seuil de détection

	Mat image_arriere_plan;

	image_arriere_plan = extraction_arriere_plan(nom_video, nb_sequences);

	vector<Mat> images_mouvement;

	images_mouvement = detection_mouvement(nom_video, image_arriere_plan,
			seuil);

	waitKey(0);

	return 0;
}
