////************************************************************************************************************////
////                          TP3: Détection et suivi de mouvement	          		   				    	    ////
////                         																	      			////
////																											////
////             Author: COULIBALY Adama                             											////
////																											////
////             Compilation: 1- make      												 						////
////						  2- ./suivi_mouvement nom_video nb_sequences seuil_detection seuil_correspondance	////
////																											////
////		     Description: Ce programme permet de suivre les mouvements des objets dans une vidéo  			////
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

//définition d'unstructure de donnée pour les boites engloba
typedef struct {
	//Point 1 du rectangle entourant l'objet
	int x1;
	int y1;
	//Point 2 du rectangle entourant l'objet
	int x2;
	int y2;
	//Variable permettant de distinguer des objets differents
	int indice;
} Boite_Englobante;

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

				// Enregistrement de l'image dans le vecteur
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

Mat amelioration_dection(Mat &image_mouvement) {

	Mat image1 = image_mouvement.clone();

	Mat img_traitee = image_mouvement.clone();
	vector<vector<Point> > contours;

	// Erosion + dilatation

	Mat element_erosion = getStructuringElement(MORPH_RECT, Size(5, 5),
			Point(1, 1));
	Mat element_dilatation = getStructuringElement(MORPH_RECT, Size(3, 3),
			Point(1, 1));

	erode(img_traitee, img_traitee, element_erosion);
	dilate(img_traitee, img_traitee, element_dilatation);

	return image_mouvement;
}

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
		boite_englobante.indice = -1;
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

// Fonction pour dessiner des cercles sur l'image
void dessiner_cercle(Mat &image_suivi, Point centre, int d,
		const Scalar& color) {
	circle(image_suivi, centre, d, color, 1, 0);
}

// Fonction pour dessiner des carrés sur l'image
void dessiner_carre(Mat &image_suivi, Point centre, int d,
		const Scalar& color) {
	rectangle(image_suivi, Point(centre.x - d, centre.y - d),
			Point(centre.x + d, centre.y + d), color, 1, 0);
}

// Fonction pour dessiner des croix sur l'image
void dessiner_croix(Mat &image_suivi, Point centre, int d,
		const Scalar& color) {
	//La forme est x pour chaque point
	line(image_suivi, Point(centre.x - d, centre.y - d),
			Point(centre.x + d, centre.y + d), color, 1, 0);
	line(image_suivi, Point(centre.x + d, centre.y - d),
			Point(centre.x - d, centre.y + d), color, 1, 0);
}

// Initialiser le Filtre de kalman
void initialiser_filtre_kalman(map<int, KalmanFilter> &liste_filtres_kalman,
		vector<Boite_Englobante> &vecteur_boites_englobates, int width,
		int height, string nom_video) {
	int maxIndex = -1;

	//Determiner l'index maximal de l'objet dans la liste
	for (int i = 0; i < (int) vecteur_boites_englobates.size(); i++) {
		if (maxIndex < vecteur_boites_englobates[i].indice
				&& vecteur_boites_englobates[i].indice != -1) {
			maxIndex = vecteur_boites_englobates[i].indice;
		}
	}

	Mat mesure = Mat::zeros(4, 1, CV_32FC1);

	//Declarer une image pour contenir tous les traces de tous les objets en mouvement
	stringstream ss;
	ss << "images_suivi/" << nom_video << ".png";
	string fileName = ss.str();
	Mat imgSuiviMouvement = imread(fileName, -1);

	for (int i = 0; i < (int) vecteur_boites_englobates.size(); i++) {

		// Initialiser le filter Kalman
		KalmanFilter filtre_kalman(4, 4, 0);

		// Initialiser des matrices
		setIdentity(filtre_kalman.transitionMatrix, cvRealScalar(1));
		filtre_kalman.transitionMatrix.at<float>(0, 2) = 1;
		filtre_kalman.transitionMatrix.at<float>(1, 3) = 1;
		setIdentity(filtre_kalman.processNoiseCov, cvRealScalar(0));
		setIdentity(filtre_kalman.measurementNoiseCov, cvRealScalar(0));
		setIdentity(filtre_kalman.measurementMatrix, cvRealScalar(1));
		setIdentity(filtre_kalman.errorCovPost, cvRealScalar(1));

		// Faire la prediction sans se baser sur les donnees historiques
		Mat predictionMat = filtre_kalman.predict();

		// Mesure
		mesure.at<float>(0, 0) = (vecteur_boites_englobates[i].x2
				+ vecteur_boites_englobates[i].x1) / 2;
		mesure.at<float>(1, 0) = (vecteur_boites_englobates[i].y2
				+ vecteur_boites_englobates[i].y1) / 2;
		// La vitesse vx, vy
		mesure.at<float>(2, 0) = 0;
		mesure.at<float>(3, 0) = 0;

		//Correction des mesures
		Mat correctionMat = filtre_kalman.correct(mesure);

		//Pour chaque objet, initialiser une image suivi
		Mat imgSuivi = Mat::zeros(height, width, CV_8UC3);

		if (vecteur_boites_englobates[i].indice == -1) {
			// Initialiser l'indice de l'objet
			maxIndex++;
			vecteur_boites_englobates[i].indice = maxIndex;

			//Dessiner le trajectoire de prediction
			dessiner_croix(imgSuivi,
					Point(predictionMat.at<float>(0, 0),
							predictionMat.at<float>(1, 0)), 3,
					CV_RGB(0, 0, 255));
			dessiner_croix(imgSuiviMouvement,
					Point(predictionMat.at<float>(0, 0),
							predictionMat.at<float>(1, 0)), 3,
					CV_RGB(0, 0, 255));

			//Dessiner le trajectoire de mesure
			dessiner_cercle(imgSuivi,
					Point(mesure.at<float>(0, 0), mesure.at<float>(1, 0)), 3,
					CV_RGB(0, 255, 0));
			dessiner_cercle(imgSuiviMouvement,
					Point(mesure.at<float>(0, 0), mesure.at<float>(1, 0)), 3,
					CV_RGB(0, 255, 0));

			//Dessiner le trajectoire de correction
			dessiner_carre(imgSuivi,
					Point(correctionMat.at<float>(0, 0),
							correctionMat.at<float>(1, 0)), 3,
					CV_RGB(255, 0, 0));
			dessiner_carre(imgSuiviMouvement,
					Point(correctionMat.at<float>(0, 0),
							correctionMat.at<float>(1, 0)), 3,
					CV_RGB(255, 0, 0));

			//Dessiner des textes dans l'image de suivi de mouvement
			stringstream ssText;
			ssText << vecteur_boites_englobates[i].indice;
			string text = ssText.str();
			int fontFace = CV_FONT_HERSHEY_SIMPLEX;
			double fontScale = 0.5;
			int thickness = 1;

			Point textPosition(mesure.at<float>(0, 0), mesure.at<float>(1, 0));
			putText(imgSuivi, text, textPosition, fontFace, fontScale,
					CV_RGB(255, 255, 255), thickness, 8);

			//Enregistrer l'image suivi pour cet objet
			stringstream ss;
			ss << "images_suivi/" << nom_video << "_objet_"
					<< vecteur_boites_englobates[i].indice << ".png";
			string filename = ss.str();

			imwrite(filename, imgSuivi);

			//Ajouter le filtre kalman actuel à la liste des filtres
			liste_filtres_kalman[vecteur_boites_englobates[i].indice] =
					filtre_kalman;
		}

	}
	imwrite(fileName, imgSuiviMouvement);
}

// Fonction pour comparer les objets d'un frame précédent et du frame courant
int chercherIndiceObjet(Boite_Englobante objet_precedent,
		vector<Boite_Englobante> &vecteur_objets_actuels,
		int seuil_correspondance) {
	// Initialiser l'indice de l'objet à -1: c'est à dire ne correspondant à aucun objet
	int indice = -1;

	float minDist = 1000000000.0;
	Boite_Englobante objet_considere;
	int centreX1 = (objet_precedent.x1 + objet_precedent.x2) / 2;
	int centreY1 = (objet_precedent.y1 + objet_precedent.y2) / 2;

	for (int i = 0; i < (int) vecteur_objets_actuels.size(); i++) {
		int centreX = (vecteur_objets_actuels[i].x1
				+ vecteur_objets_actuels[i].x2) / 2;
		int centreY = (vecteur_objets_actuels[i].y1
				+ vecteur_objets_actuels[i].y2) / 2;

		// verifier la distance entre les objets: objet précedent et chaque objet de la liste
		float distance = sqrt(
				(centreX - centreX1) * (centreX - centreX1)
						+ (centreY - centreY1) * (centreY - centreY1));
		if (distance < minDist && distance < (float) seuil_correspondance) {
			minDist = distance;
			indice = i;
			objet_considere = vecteur_objets_actuels[i];
			//listObjetsActuels[i].indice=objetPrecedent.indice;
		}
		// En cas d'egalité verifier la surface la plus petite

	}
	return indice;
}

void detection_suivi_mouvement(string nom_video, Mat image_arriere_plan,
		int seuil_detection1, int seuil_correspondance) {

	vector<Mat> images_video;
	vector<Mat> images_mouvement;
	stringstream chemin_video;
	Mat imageSuivi;
	//imageMouv  = Mat::zeros(frame.size(), CV_8UC1);

	chemin_video << "videos/" << nom_video;
	string fileName = chemin_video.str();
	char key;

	// Vecteur contenant les objets détectés du frame precedent
	vector<Boite_Englobante> ListObjetsPrecedents;
	ListObjetsPrecedents.clear();

	// Vecteur contenant les objets détectés dans le frame courant
	vector<Boite_Englobante> ListObjetsActuel;
	ListObjetsActuel.clear();

	//Vecteur contenant tous les objets
	vector<Boite_Englobante> ListObjetsTotal;
	ListObjetsTotal.clear();

	//List des Kalmam filter des objets en mouvement dans le frame courant
	map<int, KalmanFilter> listKalmanFilter;
	listKalmanFilter.clear();

	namedWindow(nom_video, CV_WINDOW_AUTOSIZE);
	namedWindow("Image Arriere Plan", CV_WINDOW_AUTOSIZE);
	namedWindow("Detection Mouvement", CV_WINDOW_AUTOSIZE);

	GaussianBlur(image_arriere_plan, image_arriere_plan, Size(5, 5), 0, 0);
	// Chargement de la video
	VideoCapture videoCapture(fileName);

	if (!videoCapture.isOpened()) {
		cout << "Impossible de lire la video : " << nom_video << endl;
		exit(0);
	} else {
		// lcture des images constituant la video
		int numFrameActuel = 0;
		while (key != 'q' && key != 'Q') {

			Mat frame, frame_gray;

			videoCapture >> frame;

			if (!frame.data) {
				cout << "Fin de lecture de la video" << endl;
				break;
			} else {

				// Conversion des images en niveau de gris
				cvtColor(frame, frame_gray, CV_BGR2GRAY);

				// Enregistrement de l'image dans le vecteur d'image
				images_video.push_back(frame_gray);

				GaussianBlur(images_video.back(), images_video.back(),
						Size(5, 5), 0, 0);
				Mat difference_images;
				// Différence entre image arriere-plan et image capturée pour détecter le mouvement
				absdiff(image_arriere_plan, images_video.back(),
						difference_images);
				images_mouvement.push_back(difference_images);
				//GaussianBlur( images_mouvement.back(), images_mouvement.back(),Size( 3, 3 ), 0, 0 );
				// Seuillage binaire pour éliminer les bruits
				threshold(images_mouvement.back(), images_mouvement.back(),
						seuil_detection1, 255.0, CV_THRESH_BINARY);
				images_mouvement.back() = amelioration_dection(
						images_mouvement.back());

				//détermination des boites englobantes pour les objets détectés
								vector<Boite_Englobante> vecteur_boites_englobantes =
										determiner_boite_englobante(images_mouvement.back());
								dessiner_boite_englobante(frame, vecteur_boites_englobantes);

				if (images_mouvement.back().data) {
					// Enregistrement de l'image de mouvement
					stringstream path;
					path << "images_mouvement/" << nom_video << "seuil_"
							<< seuil_detection1 << "_frame_"
							<< images_video.size() - 1 << ".png";
					string fileName = path.str();


				//Determiner des objets se trouvant dans le frame courant
				ListObjetsActuel = determiner_boite_englobante(
						images_mouvement.back());

				//Dessiner des boites sur l'image correspondante
				dessiner_boite_englobante(frame, ListObjetsActuel);

				imageSuivi = Mat::zeros(frame.size(), CV_8UC3);

				//*****************************************  Suivi de mouvement   *************************************/
				if (numFrameActuel == 0 && (ListObjetsActuel.size() > 0)) {
					initialiser_filtre_kalman(listKalmanFilter,
							ListObjetsActuel, frame.cols, frame.rows,
							nom_video);
					ListObjetsPrecedents = ListObjetsActuel;
					ListObjetsTotal = ListObjetsActuel;
				} else

				{
					stringstream ss;
					ss << "images_suivi/" << nom_video << ".png";
					string fileName = ss.str();

					imageSuivi = imread(fileName, -1); // chargement de l'image pour tracer le parcours
					ListObjetsTotal.clear();

					for (int i = 0; i < (int) ListObjetsPrecedents.size();
							i++) {
						// 1ere Etape: Faire la prediction des positions des objets
						Mat predictionMat =
								listKalmanFilter[ListObjetsPrecedents[i].indice].predict();

						stringstream ss;
						ss << "images_suivi/" << nom_video << "_objet_"
								<< ListObjetsPrecedents[i].indice << ".png";
						string fileName = ss.str();
						Mat imgSuiviObj = imread(fileName, -1);

						// Dessiner le trajectoire de prediction
						dessiner_croix(imgSuiviObj,
								Point(predictionMat.at<float>(0, 0),
										predictionMat.at<float>(1, 0)), 3,
								CV_RGB(255, 0, 0));
						dessiner_croix(imageSuivi,
								Point(predictionMat.at<float>(0, 0),
										predictionMat.at<float>(1, 0)), 3,
								CV_RGB(255, 0, 0));

						// Chercher la correspondance entre les objets du frame précédent et ceux du frame actuel
						int correspondance = chercherIndiceObjet(
								ListObjetsPrecedents[i], ListObjetsActuel,
								seuil_correspondance);

						if (correspondance != -1) {
							ListObjetsActuel[correspondance].indice =
									ListObjetsPrecedents[i].indice;

							// 2e Etape: Mesure des positions des objets
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

							// Les coordonnées de la vitesse
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

							// Afficher et enregistrer les resultats suivi du mouvement et dessiner le trajectoire de mesure
							dessiner_cercle(imageSuivi,
									Point(mesure.at<float>(0, 0),
											mesure.at<float>(1, 0)), 3,
									CV_RGB(0, 255, 0));
							dessiner_cercle(imgSuiviObj,
									Point(mesure.at<float>(0, 0),
											mesure.at<float>(1, 0)), 3,
									CV_RGB(0, 255, 0));

							// 3e Etape: Faire la correction des positions des objets
							Mat correctionMat =
									listKalmanFilter[ListObjetsPrecedents[i].indice].correct(
											mesure);

							// Dessiner la trajectoire de la correction
							dessiner_carre(imageSuivi,
									Point(correctionMat.at<float>(0, 0),
											correctionMat.at<float>(1, 0)), 3,
									CV_RGB(0, 0, 255));
							dessiner_carre(imgSuiviObj,
									Point(correctionMat.at<float>(0, 0),
											correctionMat.at<float>(1, 0)), 3,
									CV_RGB(0, 0, 255));

							//Enregistrer l'image suivi pour cet objet
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
							frame.cols, frame.rows, nom_video);
					ListObjetsPrecedents = ListObjetsTotal;
				}

				//Ajouter les noms des objets en mouvement dans le frame courant
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

				//Affichage et enregistrement des resultats
				stringstream ss1;
				ss1 << "images_suivi/" << nom_video << ".png";
				string fileName1 = ss1.str();
				if (!imwrite(fileName1, imageSuivi))
					cout << "Erreur lors de l'enregistrement de " << fileName1
							<< endl;

						stringstream ss2;
				ss2 << "images_videos/" << nom_video << "_" << numFrameActuel
						<< ".png";
				string fileName2 = ss2.str();
				if (!imwrite(fileName2, frame))
					cout << "Erreur lors de l'enregistrement de " << fileName2
							<< endl;

				//enrégistrement de l'image de la vidéo avec les boites englobantes
								stringstream ss3;
								ss3 << "images_mouvement/" << nom_video << "seuil_" << seuil_detection1
										<< "_frame_" << images_video.size() - 1 << ".png";
								string fileName3 = ss3.str();
								if (!imwrite(fileName3, images_mouvement.back()))
									cout << "Erreur lors de l'enregistrement de " << fileName3
											<< endl;

				imshow("Suivi Mouvement", imageSuivi);

				numFrameActuel++;

				imshow(nom_video,frame);
				imshow("Image Arriere Plan", image_arriere_plan);
				imshow("Detection Mouvement", images_mouvement.back());

				key = cvWaitKey(40);

			}
		}

	}
	cvDestroyAllWindows();

}
}

//fonction principal
int main(int argc, char** argv)
{

	// Réccupération des paramètres du programme
	string nom_video = argv[1];  //Nom de la video
	int nb_sequences = atoi(argv[2]); //Nombre de séquence d'images
	int seuil_detection = atoi(argv[3]); //seuil de détection d'images
	int seuil_correspondance = atoi(argv[4]); //seuil de correspondance

	Mat image_arriere_plan;

	image_arriere_plan = extraction_arriere_plan(nom_video, nb_sequences);

	detection_suivi_mouvement(nom_video, image_arriere_plan, seuil_detection,
			seuil_correspondance);

	waitKey(0);

	return 0;
}
