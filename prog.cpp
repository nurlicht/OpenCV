#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dirent.h>
#include <string>
#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

/**
--------------------------
	Global Variables
--------------------------
*/
String imageFolder;
String* fileNames;
int nFrames;
int frameSliderMax;
int frameSlider;
int trackSlider;
int trackSliderMax;
char* windowName;
char* keyPointWindowName;

Mat* im;
Mat dtct;
vector<KeyPoint> kp;
vector<Point2f> points;



/**
--------------------------
	Class "Headers"
--------------------------
*/
class ResultsTable {
	private:
	
	public:
	ResultsTable();
	double getValue(char*, int);
};
class LoadImage {
  private:
	
	void getFileNames();
	void setIm();

  public:
	bool invalidFileNames; 

	LoadImage();
	LoadImage(String imageFolder_);
	Mat* getIm();
	int getNFrames();
	~LoadImage();
};

class Spot {
	private:

	public:
	double x;
	double y;
	
	double Dx;
	double Dy;
	
	int frame;
	
	double I;
	int spotIndex;

	int roiIndex;
	int trackIndex;

	Spot ();
	Spot (double x, double y);
	Spot (double x, double y, int frame);
	Spot (double x, double y, int frame, int spotIndex);
	Spot (double x, double y, int frame, int spotIndex, int roiIndex);
	void setD (double D);
	void setD (double Dx, double Dy);
	~Spot ();
};

class Track {
	private:
	double vNorm2(double dx, double dy);
	double vNorm(double dx, double dy);
	int db2int(double db);
    string d2s(int x);
    int s2d(String s);
	
	public:
	Spot* spots;
	int nSpots;
	int frameStart;
	int frameEnd;
	int frameNumber;
	int trackIndex;

	Track(Spot* s, int sLength);
	Track(Spot s);
	Track(int n);
	Track();
	~Track();
	void setDim(int n);
	void setFrameSpotInfo();
	void append(Spot s);
	void append(Spot* s, int nS);
	void setTrackIndex(int N);
	Spot* cloneSpots();
	double getDistance(Spot s);
	double* getDistance(Spot* s, int nS);
	void loadTrackMate1Subset(ResultsTable rt, int startIndex);
};

class TrackSet {
	private:
	ResultsTable rt;
	void setNAllSpots();
	void findNSpotsArrayFromTrackSet();
	void loadTrackMate1File(char* pathName);
	void loadTracksFromRT();
	void openFileInRT(char* pathName);
	void setObjParams();
	void closeRT();
	int db2int(double db);
    string d2s(int x);
    int s2d(String s);
	
	public:
	Track* tracks;
	int nTracks;
	int nAllSpots;
	int* nSpotsArray;
	int frameStartAll;
	int frameEndAll;
	int frameNumberAll;
	char* pathName;
	char* trackingMethod;
	int logFlag;
	
	TrackSet();
	TrackSet(int nTracks);
	TrackSet(Track* ts, int nTS);
	TrackSet(char* pathName, char* trackingMethod);
	TrackSet(char* pathName, char* trackingMethod, int logFlag);
	~TrackSet();
	void vrtlCnstrctr(char* pathName, char* trackingMethod);
	void setFrameSpotInfo();
	void findNTracksFromRT();
	void findNSpotsArrayFromRT();
	void merge (Track t);
	void merge (Track* ts, int nNew);
};

class SpotDetection {
	private:
	int nFrameSpots;
	int nAllSpots;
	int* nSpotsArray;
	int* nSpotsArrayAccumulated;
	int firstDetectedFrame;
	int lastDetectedFrame;
	int nFramesWithSpots;
	int* nSpotsArrayPure;
	Spot* allSpots;
	Spot* frameSpots;
	
	void initializeVariables();
	void updateFrameVariables();
	void appendFrameSpots();
	void findAllSpots();
	
	public:
	SpotDetection();
	int getNFrameSpots();
	int getNAllSpots();
	int* getNSpotsArray();
	int* getNSpotsArrayAccumulated();
	int getFirstDetectedFrame();
	int getLastDetectedFrame();
	int getNFramesWithSpots();
	int* getNSpotsArrayPure();
	Spot* getAllSpots();
	Spot* getFrameSpots(int n);
	~SpotDetection();
};

class Linking {
	private:
	int nRows;
	int nColumns;
	int nAll;
	double linkThreshold;
	Spot* sRow;
	Spot* sColumn;
	int* I;
	int* J;
	double* dm;
	
	void link();
	double vNorm(double dx, double dy);
	double getDistance(Spot s1, Spot s2);
	void initializeIndexArrays();
	void setDistanceMatrix();
	void maskDiagonalElements();
	void sortDistanceMatrix();
	void setDimensions(int nRows, int nColumns);
	
	public:
	int nLinks;
	int* rowIndexForColumn;
	int* columnIndexForRow;
	
	Linking();
	Linking(Spot* sRow, int nSRow, Spot* sColumn, int nSColumn, double linkThreshold);
	Linking(Spot* s, int nS, double linkThreshold);
};

class Tracking {
	private:
	String* trackingParams;
	SpotDetection sd;
	bool pairFlag = false;
	int firstDetectedFrame;
	double linkThreshold;
	Track* ts;
	Linking* linking;
	
	void initializeTrackSet();
	void linkFrames();
	Spot* getPairs(Spot* spots);
	Spot* getPreviousSpots(int frame);
	void setGlobalTrackIndices();
	void addFrameSpotsToTracks(Spot* frameSpots, int* rowIndexForColumn, int frame);
		
	public:
	TrackSet trackSet;
	Tracking(SpotDetection sd);
	Tracking(SpotDetection sd, String* trackingParams);
	~Tracking();
};

Tracking* trng;



/**
--------------------------
	Stand-alone functions (without classes)
--------------------------
*/

void setBD(bool drawKPFlag) {
	SimpleBlobDetector::Params bdParams;
	//Ptr<SimpleBlobDetector> detector;
	vector<KeyPoint>::iterator blobIterator;
	int blobCntr;

	bdParams.minThreshold = 10;
	bdParams.maxThreshold = 200;

	bdParams.minArea = 20;
	bdParams.maxArea = 1500;

	bdParams.filterByCircularity = true;
	bdParams.minCircularity = 0.2;

	#if CV_MAJOR_VERSION < 3
	SimpleBlobDetector detector(bdParams);
	detector.detect(*(im + frameSlider), kp);
	#else
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(bdParams);
	detector->detect(*(im + frameSlider), kp);
	#endif

	if (drawKPFlag) {
		drawKeypoints(*(im + frameSlider), kp, dtct, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	}		
}

void setFrame(int n) {
	frameSlider = n;
}

void createWindow() {
	windowName = "Stack Viewer";
	keyPointWindowName = "Key Point";
	namedWindow(windowName, WINDOW_NORMAL);
}

void on_trackbar( int, void* )
{
	bool showKPFlag = true;
	setBD(showKPFlag);
	if (showKPFlag) {
		imshow(windowName, dtct);
		waitKey(1);
	}
}

void on_trackSlider( int, void* )
{
	int x;
	int y;
	int r = 6;
	int cntr;
	int nSpots;
	int nTracks;
	bool showKPFlag = true;
	
	if (showKPFlag && (trackSlider < (trng->trackSet).nTracks)) {
		on_trackbar(0, 0);
		nTracks = (trng->trackSet).nTracks;
		nSpots = (trng->trackSet).tracks[trackSlider].nSpots;
		cout << "nTracks = " << nTracks << "\n";
		cout << "Selected Track: " << trackSlider << " with " << nSpots << " spots\n";
		
		for (cntr = 0; cntr < nSpots; cntr++) {
			x = (trng->trackSet).tracks[trackSlider].spots[cntr].x;
			y = (trng->trackSet).tracks[trackSlider].spots[cntr].y;
			cout << "x = " << x << ", y = " << y << "\n";
			circle(dtct, Point(x, y), r, Scalar(255, 0, 0), -1);
		}
		imshow(windowName, dtct);
		cout << "Window updated with Mat dtct.\n";
		waitKey(1);
	}
}


void initializeTrackBar() {
	frameSlider = 0;
	frameSliderMax = nFrames - 1;
	createTrackbar( "Frame", windowName, &frameSlider,  frameSliderMax, on_trackbar );
}



/**
--------------------------
	The main() function
--------------------------
*/


int main( )
{
	LoadImage li = LoadImage("./Images/");
	if (li.invalidFileNames) {
		return -1;
	}
	
	createWindow();
	imshow(windowName, *(im + frameSlider));
	initializeTrackBar();
	on_trackbar( 0, 0 );
	SpotDetection sd = SpotDetection();
	waitKey(0);
	
	trng = new Tracking(sd);
	waitKey(0);
	
	trackSlider = 0;
	trackSliderMax = (trng->trackSet).nTracks - 1;
	createTrackbar( "Track", windowName, &trackSlider,  trackSliderMax, on_trackSlider );
	waitKey(0);
	
	delete trng;
	return 0;
}



/**
--------------------------
	Class definitions
--------------------------
*/

ResultsTable::ResultsTable() {
}

double ResultsTable::getValue(char* columnHeader, int rowIndex) {
	double a;
	a = 4;
	return a;
}




LoadImage::LoadImage() {
	imageFolder = "/home/ali/opencv/myProj/4_StackViewer/B/";
	getFileNames();
	setIm();
}

LoadImage::LoadImage(String imageFolder_) {
	imageFolder = imageFolder_;
	getFileNames();
	setIm();
}

void LoadImage::getFileNames() {
 DIR *dir; 
 struct dirent *ent;
 int nEntities;
 int frameCntr;
 char *temp;
 String A;
 int cntr;

 A = {"A"};
 if ((dir = opendir(imageFolder.c_str())) != NULL) {
   while ((ent = readdir(dir)) != NULL) {
     temp = ent->d_name;
     if (temp[0] == A[0]) {
       nFrames++;
     }
     nEntities++;
   }
   closedir(dir);
 }
	
	
 if (nFrames) {
   frameCntr = 0;
   fileNames = new String[nFrames];
   dir = opendir(imageFolder.c_str());
   for (cntr = 0; cntr < nEntities; cntr++) {
     ent = readdir(dir);
     temp = ent->d_name;
     if (temp[0] == A[0]) {
       fileNames[frameCntr] = ent->d_name;
       cout << "File #" << (frameCntr + 1) << " is " << fileNames[frameCntr] << "\n";
       frameCntr++;
     }
   }
   closedir(dir);

   cout << "\n" << "After sorting:" <<  "\n\n";
   sort(fileNames, fileNames + nFrames);
   for (cntr = 0; cntr < nFrames; cntr++) {
       cout << "File #" << (cntr + 1) << " is " << fileNames[cntr] << "\n";
   }
   //delete [] fileNames;
   invalidFileNames = false;
 } else {
   invalidFileNames = true;
 }
 return;	
}

void LoadImage::setIm() {
	int cntr;
	
	if (invalidFileNames || (nFrames == 0)) {
		return;
		cout << "No image was loaded: nFrames = " << nFrames << ", invalidFileNames = " << invalidFileNames << ".\n";
	}
	
	im = new Mat[nFrames];
	for (cntr = 0; cntr < nFrames; cntr++) {
		*(im + cntr) = imread(imageFolder + fileNames[cntr],IMREAD_GRAYSCALE);
	}
	cout << "All " << nFrames << " images were loaded.\n\n";
}

Mat* LoadImage::getIm() {
	return im;
}

int LoadImage::getNFrames() {
	return nFrames;
}

LoadImage::~LoadImage() {
	delete [] im;
}




SpotDetection::SpotDetection() {
	initializeVariables();
	findAllSpots();
	cout << "\nA total of " << nAllSpots << " spots in " << nFrames << " frames were detected.\n";
}

void SpotDetection::updateFrameVariables() {
	vector<KeyPoint>::iterator bi;
	int cntr;
	Spot s;

	nFrameSpots = 0;
	for (bi = kp.begin(); bi != kp.end(); bi++) {
		nFrameSpots++;
	}
	
	nSpotsArray[frameSlider] = nFrameSpots;
	nSpotsArrayAccumulated[frameSlider] = (frameSlider == 0) ? nFrameSpots : (nFrameSpots + nSpotsArrayAccumulated[frameSlider - 1]);
	frameSpots = NULL;
	if (nFrameSpots == 0) {
		return;
	} else {
		nSpotsArrayPure[nFramesWithSpots++] = nFrameSpots;
		if (firstDetectedFrame == -1) {
			firstDetectedFrame = frameSlider + 1;
		}
		lastDetectedFrame = frameSlider + 1;
	}
	
	frameSpots = new Spot[nFrameSpots];
	cout << "Spot detection (frame #" << (frameSlider + 1) << " with " << nFrameSpots << " spots):\n";
	
	for (cntr = 0; cntr < nFrameSpots; cntr++) {
		bi = kp.begin() + cntr;
		s = Spot(bi->pt.x, bi->pt.y, frameSlider + 1);
		s.setD(bi->size);
		*(frameSpots + cntr) = s;
		cout << "Spot #" << cntr << ": (" << s.x << "," << s.y << ")\n";
	}
	cout << "\n";
}

void SpotDetection::appendFrameSpots() {
	int cntr;
	Spot* temp;
	
	if (nFrameSpots == 0) {
		return;
	}
	
	temp = new Spot[nAllSpots];
	for (cntr = 0; cntr < nAllSpots; cntr++) {
		temp[cntr] = allSpots[cntr];
	}
	
	allSpots = new Spot[nAllSpots + nFrameSpots];
	for (cntr = 0; cntr < nAllSpots; cntr++) {
		allSpots[cntr] = temp[cntr];
	}
	for (cntr = 0; cntr < nFrameSpots; cntr++) {
		allSpots[nAllSpots + cntr] = frameSpots[cntr];
	}
	nAllSpots += nFrameSpots;

	delete [] temp;
}

void SpotDetection::findAllSpots() {
	int cntr;
	bool showKPFlag;
	
	showKPFlag = false;
	
	for (cntr = 0; cntr < nFrames; cntr++) {
		setFrame(cntr);
		setBD(showKPFlag);
		updateFrameVariables();
		appendFrameSpots();
	}
}
	
void SpotDetection::initializeVariables() {
	nAllSpots = 0;
	firstDetectedFrame = -1;
	lastDetectedFrame = -1;
	nFramesWithSpots = 0;
	nSpotsArray = new int[nFrames];
	nSpotsArrayAccumulated = new int[nFrames];
	nSpotsArrayPure = new int[nFrames];
}		 

int SpotDetection::getNFrameSpots() {
	return nFrameSpots;
}

int SpotDetection::getNAllSpots() {
	return nAllSpots;
}

int* SpotDetection::getNSpotsArray() {
	return nSpotsArray;
}

int* SpotDetection::getNSpotsArrayAccumulated() {
	return nSpotsArrayAccumulated;
}

int SpotDetection::getFirstDetectedFrame() {
	return firstDetectedFrame;
}

int SpotDetection::getLastDetectedFrame() {
	return lastDetectedFrame;
}

int SpotDetection::getNFramesWithSpots() {
	return nFramesWithSpots;
}

int* SpotDetection::getNSpotsArrayPure() {
	return nSpotsArrayPure;
}

Spot* SpotDetection::getAllSpots() {
	return allSpots;
}

Spot* SpotDetection::getFrameSpots(int n) {
	int startIndex;
	int cntr;
	
	frameSpots = NULL;
	nFrameSpots = nSpotsArray[n - 1];
	if (nFrameSpots > 0) {
		frameSpots = new Spot[nFrameSpots];
		startIndex = (n > 1) ? nSpotsArrayAccumulated[n - 2] : 0;
		for (cntr = 0; cntr < nFrameSpots; cntr++) {
			frameSpots[cntr] = allSpots[cntr + startIndex];
		}
	}
	cout << "Frame #" << n << ": startIndex = " << startIndex << "\n";
	return frameSpots;
}

SpotDetection::~SpotDetection() {
}



Linking::Linking() {
}

Linking::Linking(Spot* sRow, int nSRow, Spot* sColumn, int nSColumn, double linkThreshold) {
	this->sRow = sRow;
	this->sColumn = sColumn;
	this->linkThreshold = linkThreshold;
	setDimensions(nSRow, nSColumn);
	initializeIndexArrays();
	setDistanceMatrix();
	sortDistanceMatrix();
	link();

	int cntr;
	cout << "sColumn: ";
	for (cntr = 0; cntr < nSColumn; cntr++) {
		cout << "(" << (sColumn + cntr)->x << "," << (sColumn + cntr)->y << "), ";
	}
	cout << "\n";
	cout << "sRow: ";
	for (cntr = 0; cntr < nSRow; cntr++) {
		cout << "(" << (sRow + cntr)->x << "," << (sRow + cntr)->y << "), ";
	}
	cout << "\n\n";
}

Linking::Linking(Spot* s, int nS, double linkThreshold) {
	this->sRow = s;
	this->sColumn = s;
	this->linkThreshold = linkThreshold;
	setDimensions(nS, nS);
	initializeIndexArrays();
	setDistanceMatrix();
	maskDiagonalElements();
	sortDistanceMatrix();
	link();
}

void Linking::link() {
	int cntr;
	int i;
	int j;
	
	nLinks = 0;
	for (cntr = 0; (cntr < nAll) && (nLinks < nColumns); cntr++) {
		j = J[cntr];
		i = I[cntr];
		if ((columnIndexForRow[i] == -1) && (rowIndexForColumn[j] == -1)) {
			if (dm[cntr] <= linkThreshold) {
				columnIndexForRow[i] = j;
				rowIndexForColumn[j] = i;
				nLinks++;
			}
		}
	}
	//cout << "Linking: " << nLinks << " links (to previous tracks) and " << (nColumns - nLinks) << " new tracks\n";
}

void Linking::sortDistanceMatrix() {
	int mCntr1;
	int mCntr2;
	int dummy;
	double dummy_;
	
	for (mCntr1 = 0; mCntr1 < (nAll - 1); mCntr1++) {
		for (mCntr2 = (mCntr1 + 1); mCntr2 < nAll; mCntr2++) {
			if (dm[mCntr2] < dm[mCntr1]) {
				dummy_ = dm[mCntr2];
				dm[mCntr2] = dm[mCntr1];
				dm[mCntr1] = dummy_;

				dummy = I[mCntr2];
				I[mCntr2] = I[mCntr1];
				I[mCntr1] = dummy;

				dummy = J[mCntr2];
				J[mCntr2] = J[mCntr1];
				J[mCntr1] = dummy;
			}
		}
	}
}

void Linking::maskDiagonalElements() {
	int rowCntr;
	int columnCntr;
	int cntr;
	int maxtrixCntr;
	double dmMax;

	dmMax = dm[0];
	for (cntr = 1; cntr < nAll; cntr++) {
		if (dmMax < dm[cntr]) {
			dmMax = dm[cntr];
		}
	}
	dmMax += 1;

	maxtrixCntr = 0;
	for (rowCntr = 0; rowCntr < nRows; rowCntr++) {
		for (columnCntr = 0; columnCntr < nColumns; columnCntr++) {
			if (rowCntr == columnCntr) {
				dm[maxtrixCntr] = dmMax;
			}
			maxtrixCntr++;
		}
	}
}

void Linking::setDistanceMatrix() {
	int rowCntr;
	int columnCntr;
	int maxtrixCntr;

	maxtrixCntr = 0;
	for (rowCntr = 0; rowCntr < nRows; rowCntr++) {
		for (columnCntr = 0; columnCntr < nColumns; columnCntr++) {
			I[maxtrixCntr] = rowCntr;
			J[maxtrixCntr] = columnCntr;
			dm[maxtrixCntr++] = getDistance(sRow[rowCntr], sColumn[columnCntr]);
		}
	}
}

void Linking::setDimensions(int nRows, int nColumns) {
	int cntr;
	
	this->nRows = nRows;
	this->nColumns = nColumns;
	this->nAll = nRows * nColumns;
	dm = new double[nAll];
	I = new int[nAll];
	J = new int[nAll];
	columnIndexForRow = new int[nRows];
	rowIndexForColumn = new int[nColumns];
}
	
void Linking::initializeIndexArrays() {
	int cntr;

	for (cntr = 0; cntr < nRows; cntr++) {
		columnIndexForRow[cntr] = -1;
	}
	for (cntr = 0; cntr < nColumns; cntr++) {
		rowIndexForColumn[cntr] = -1;
	}
}

double Linking::getDistance(Spot s1, Spot s2) {
	return vNorm(s1.x - s2.x, s1.y - s2.y);
}

double Linking::vNorm(double dx, double dy) {
	return sqrt(dx * dx + dy * dy);
}




Tracking::Tracking(SpotDetection sd) {
	linkThreshold = 100.0;
	this->sd = sd;
	initializeTrackSet();
	linkFrames();
	cout << trackSet.nTracks << " tracks were detected.\n";
}

Tracking::Tracking(SpotDetection sd, String* trackingParams) {
	linkThreshold = 100.0;
	this->sd = sd;
	this->trackingParams = trackingParams;
	initializeTrackSet();
	linkFrames();
	cout << trackSet.nTracks << " tracks were detected.\n";
}

void Tracking::initializeTrackSet() {
	int nFirstTracks;
	int cntr;
	Spot* firstSpots;

	if (firstDetectedFrame == -1) {
		trackSet = NULL;
		return;
	}

	firstSpots = sd.getFrameSpots(sd.getFirstDetectedFrame());
	if (pairFlag) {
		firstSpots = getPairs(firstSpots);
	}
	nFirstTracks = (sd.getNSpotsArray())[sd.getFirstDetectedFrame()];
	trackSet = TrackSet(nFirstTracks);
	for (cntr = 0; cntr < nFirstTracks; cntr++) {
		*(trackSet.tracks + cntr) = Track(firstSpots[cntr]);
	}
}

void Tracking::linkFrames() {
	Spot* frameSpots;
	Spot* previousSpots;
	int nFrameSpots;
	int nPrFrameSpots;
	int frameCntr;
	int frameCntrPure;
	int secondPureFrameInf;

	secondPureFrameInf = sd.getFirstDetectedFrame() + 1;
	cout << "sd.getFirstDetectedFrame() = " << sd.getFirstDetectedFrame() << ", secondPureFrameInf = " << secondPureFrameInf << "\n\n";
	frameCntrPure = 1;
	for (frameCntr = secondPureFrameInf; frameCntr <= nFrames; frameCntr++) {
		nFrameSpots = (sd.getNSpotsArray())[frameCntr - 1];
		cout << "Tracking::linkFrames --> frameCntr = " << frameCntr << ", nFrameSpots = " << nFrameSpots << "\n";
		if (nFrameSpots > 0) {
			frameSpots = sd.getFrameSpots(frameCntr);
			previousSpots = getPreviousSpots(frameCntr);
			nPrFrameSpots = (sd.getNSpotsArray())[frameCntr - 2];
			if (pairFlag) {
				frameSpots = getPairs(frameSpots);
				previousSpots = getPairs(previousSpots);
			}
			linking = new Linking(previousSpots, nPrFrameSpots, frameSpots, nFrameSpots, linkThreshold);
			addFrameSpotsToTracks(frameSpots, linking->rowIndexForColumn, frameCntr);
			delete [] linking;
			frameCntrPure++;
		}
	}
	setGlobalTrackIndices();
}

Spot* Tracking::getPairs(Spot* spots) {
	Spot* pairs;
	/*
	*/
	return pairs;
}

Spot* Tracking::getPreviousSpots(int frame) {
	return ((frame == 1) ? NULL : sd.getFrameSpots(frame - 1));
}

void Tracking::setGlobalTrackIndices() {
	int trackCntr;
	int nTracks;
	int nSpots;

	nTracks = trackSet.nTracks;
	for (trackCntr = 0; trackCntr < nTracks; trackCntr++) {
		nSpots = trackSet.tracks[trackCntr].nSpots;
		trackSet.tracks[trackCntr].frameStart = trackSet.tracks[trackCntr].spots[0].frame;
		trackSet.tracks[trackCntr].frameEnd = trackSet.tracks[trackCntr].spots[nSpots - 1].frame;
		trackSet.tracks[trackCntr].setTrackIndex(trackCntr);
	}
}

void Tracking::addFrameSpotsToTracks(Spot* frameSpots, int* rowIndexForColumn, int frame) {
	int cntr;
	int nFrameSpots;

	nFrameSpots = sd.getNSpotsArray()[frame];
	for (cntr = 0; cntr < nFrameSpots; cntr++) {
		if (rowIndexForColumn[cntr] != -1) {
			trackSet.tracks[rowIndexForColumn[cntr]].append(frameSpots[cntr]);
		} else {
			trackSet.merge(Track(frameSpots[cntr]));
		}
	}
}

Tracking::~Tracking() {
	//delete [] linking;
}





Spot::Spot () {
}

Spot::Spot (double x, double y) {
	this->x = x;
	this->y = y;
}

Spot::Spot (double x, double y, int frame) {
	this->x = x;
	this->y = y;
	this->frame = frame;
}

Spot::Spot (double x, double y, int frame, int spotIndex) {
	this->x = x;
	this->y = y;
	this->frame = frame;
	this->spotIndex = spotIndex;
}

Spot::Spot (double x, double y, int frame, int spotIndex, int roiIndex) {
	this->x = x;
	this->y = y;
	this->frame = frame;
	this->spotIndex = spotIndex;
	this->roiIndex = roiIndex;
}

void Spot::setD (double D) {
	Dx = D;
	Dy = D;
}

void Spot::setD (double Dx, double Dy) {
	this->Dx = Dx;
	this->Dy = Dy;
}

Spot::~Spot () {
}





Track::Track() {
	setDim(0);
}

Track::Track(int n) {
	setDim(n);
}

Track::Track(Spot s) {
	setDim(1);
	spots[0] = s;
}

Track::Track(Spot* s, int sLength) {
	setDim(sLength);
	spots = s;
}

Track::~Track() {
}

void Track::setDim(int n) {
	nSpots = n;
	spots = new Spot[n];
}

void Track::setFrameSpotInfo() {
	if (spots == NULL) {
		cout << "Error: Track.spots = NULL!\n";
		return;
	}
	frameStart = spots[0].frame;
	frameEnd = spots[nSpots - 1].frame;
	frameNumber = frameEnd - frameStart + 1;
}

void Track::append(Spot s) {
	int cntr;
	Spot* spots_;

	spots_ = new Spot[nSpots + 1];
	for (cntr = 0; cntr < nSpots; cntr++) {
		spots_[cntr] = spots[cntr];
	}
	spots_[nSpots++] = s;
	spots = spots_;
}

void Track::append(Spot* s, int nS) {
	int cntr;
	for (cntr = 0; cntr < nS; cntr++) {
		append(s[cntr]);
	}
}

void Track::setTrackIndex(int N) {
	int cntr;

	trackIndex = N;
	for (cntr= 0; cntr < nSpots; cntr++) {
		spots[cntr].trackIndex = N;
	}
}

Spot* Track::cloneSpots() {
	Spot* sp = new Spot[nSpots];
	int cntr;
	for (cntr = 0; cntr < nSpots; cntr++) {
		sp[cntr] = spots[cntr];
	}
	return sp;
}

double* Track::getDistance(Spot* s, int nS) {
	int cntr;
	double* distances = new double[nS];
	for (cntr = 0; cntr < nS; cntr++) {
		distances[cntr] = getDistance(s[cntr]);
	}
	return distances;
}

double Track::vNorm2(double dx, double dy) {
	return (dx * dx + dy * dy);
}

double Track::vNorm(double dx, double dy) {
	return sqrt(dx * dx + dy * dy);
}

double Track::getDistance(Spot s) {
	return vNorm(spots[nSpots - 1].x - s.x, spots[nSpots - 1].y - s.y);
}

void Track::loadTrackMate1Subset(ResultsTable rt, int startIndex) {
	int spotCntr;
	double x;
	double y;
	int frame;
	for (spotCntr = 0; spotCntr < nSpots; spotCntr++) {
		x = rt.getValue("POSITION_X", spotCntr + startIndex);
		y = rt.getValue("POSITION_Y", spotCntr + startIndex);
		frame = db2int(rt.getValue("FRAME", spotCntr + startIndex));
		*(spots + spotCntr) = Spot(x, y, frame, spotCntr);
	}
	setFrameSpotInfo();
}

int Track::db2int(double db) {
	return (int) db;
}

string Track::d2s(int x) {
	ostringstream stream;
	stream << x;
	return stream.str();
}

int Track::s2d(String s) {
	istringstream stream(s);
	int x;
	if (!(stream >> x)) {
		x = 0;
	}
	return x;
}





TrackSet::TrackSet() {
	this->nTracks = 0;
	this->tracks = NULL;
}

TrackSet::TrackSet(int nTracks) {
	this->nTracks = nTracks;
	this->tracks = new Track[nTracks];
}

TrackSet::TrackSet(Track* ts, int nTS) {
	tracks = ts;
	nTracks = nTS;
	findNSpotsArrayFromTrackSet();
	setNAllSpots();
}

TrackSet::TrackSet(char* pathName, char* trackingMethod) {
	vrtlCnstrctr(pathName, trackingMethod);
}

TrackSet::TrackSet(char* pathName, char* trackingMethod, int logFlag) {
	vrtlCnstrctr(pathName, trackingMethod);
	this->logFlag = logFlag;
}

TrackSet::~TrackSet() {
}

int TrackSet::db2int(double db) {
	return (int) db;
}

void TrackSet::findNSpotsArrayFromRT() {
	int cntr;
	int currentTrackID = db2int(rt.getValue("TRACK_ID", 0));
	int lastTrackID = currentTrackID;
	int currentTrackIndex = 0;
	int nCurrentTrack = 1;
	for (cntr = 1; cntr < nAllSpots; cntr++) {
		currentTrackID = db2int(rt.getValue("TRACK_ID", cntr));
		if (currentTrackID != lastTrackID) {
			nSpotsArray[currentTrackIndex++] = nCurrentTrack;
			lastTrackID = currentTrackID;
			nCurrentTrack = 1;
		} else {
			nCurrentTrack++;
		}
	}
	if (currentTrackID == lastTrackID) {
		nSpotsArray[currentTrackIndex++] = nCurrentTrack;
	}
}

void TrackSet::findNTracksFromRT() {
	int cntr;
	int currentTrackID;
	int lastTrackID;

	lastTrackID = db2int(rt.getValue("TRACK_ID", 0));
	nTracks = 1;
	for (cntr = 1; cntr < nAllSpots; cntr++) {
		currentTrackID = db2int(rt.getValue("TRACK_ID", cntr));
		if (currentTrackID != lastTrackID) {
			lastTrackID = currentTrackID;
			nTracks++;
		}
	}
}

void TrackSet::setFrameSpotInfo() {
	frameStartAll = tracks[0].frameStart;
	frameEndAll = tracks[nTracks - 1].frameEnd;
	frameNumberAll = frameEndAll - frameStartAll + 1;
}

void TrackSet::loadTracksFromRT() {
	int trackCntr;
	int nSpotCurrentTrack;
	int startIndex;

	if (nTracks == 0) {
		return;
	}
	startIndex = 0;
	for (trackCntr = 0; trackCntr < nTracks; trackCntr++) {
		nSpotCurrentTrack = nSpotsArray[trackCntr];
		tracks[trackCntr] = Track(nSpotCurrentTrack);
		tracks[trackCntr].loadTrackMate1Subset(rt, startIndex);
		tracks[trackCntr].setFrameSpotInfo();
		tracks[trackCntr].trackIndex = trackCntr;
		startIndex += nSpotsArray[trackCntr];
	}
	if (logFlag) {
		//IJ.log("nTrack = " + d2s(nTracks) + "; nAllSpots = " + d2s(nAllSpots) + "");
	}
}

void TrackSet::openFileInRT(char* pathName) {
	closeRT();
}

void TrackSet::setObjParams() {
	//nAllSpots = rt.size();
	if (nAllSpots == 0) {
		nTracks = 0;
		return;
	}
	findNTracksFromRT();
	findNSpotsArrayFromRT();
	tracks = new Track[nTracks];
}

void TrackSet::closeRT() {
	/*
	*/
}

void TrackSet::loadTrackMate1File(char* pathName) {
	openFileInRT(pathName);
	setObjParams();
	loadTracksFromRT();
	closeRT();
}

void TrackSet::vrtlCnstrctr(char* pathName, char* trackingMethod) {
	this->pathName = pathName;
	this->trackingMethod = trackingMethod;
	logFlag = true;
	//trackingMethod = trackingMethod.toLowerCase();
	if (trackingMethod == "trackmate1") {
		this->loadTrackMate1File(pathName);
		this->setFrameSpotInfo();
		//IJ.log("In 'Tracks()': pathName = " + pathName);
	} else if (trackingMethod == "realtime1") {
	} else {
	}
}

string TrackSet::d2s(int x) {
	ostringstream stream;
	stream << x;
	return stream.str();
}

int TrackSet::s2d(String s) {
	istringstream stream(s);
	int x;
	if (!(stream >> x)) {
		x = 0;
	}
	return x;
}

void TrackSet::setNAllSpots() {
	int cntr;

	nAllSpots = 0;
	for (cntr = 0; cntr < nTracks; cntr++) {
		nAllSpots += nSpotsArray[cntr];
	}
}

void TrackSet::findNSpotsArrayFromTrackSet() {
	int cntr;

	nSpotsArray = new int[nTracks];
	for (cntr = 0; cntr < nTracks; cntr++) {
		*(nSpotsArray + cntr) = tracks[cntr].nSpots;
	}
}

void TrackSet::merge (Track t) {
	int cntr;
	int nNew;

	nNew = t.nSpots;
	if (nNew == 0) {
		return;
	}

	Track* tracks_ = new Track[nTracks + 1];
	for (cntr = 0; cntr < nTracks; cntr++) {
		tracks_[cntr] = tracks[cntr];
	}
	tracks_[nTracks] = t;
	tracks = new Track[nTracks + 1];
	tracks = tracks_;
	nTracks ++;
	findNSpotsArrayFromTrackSet();
	setNAllSpots();
}

void TrackSet::merge (Track* ts, int nNew) {
	int cntr;

	if (nNew == 0) {
		return;
	}

	Track* tracksOriginal = new Track[nTracks];
	for (cntr = 0; cntr < nTracks; cntr++) {
		tracksOriginal[cntr] = tracks[cntr];
	}
	tracks = new Track[nTracks + nNew];
	for (cntr = 0; cntr < nTracks; cntr++) {
		tracks[cntr] = tracksOriginal[cntr];
	}
	for (cntr = 0; cntr < nNew; cntr++) {
		tracks[cntr + nTracks] = ts[cntr];
	}
	nTracks += nNew;
	findNSpotsArrayFromTrackSet();
	setNAllSpots();
}
