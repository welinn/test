#include <stdio.h>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;

Mat sample(Mat, int, int);
void onMouse(int, int, int, int, void*);
void onMouseTowing(int, int, int, int, void*);
Point min(Point, Point);

int maskSize = 5, trai = 0;
int count = 10;
float *trainingData;
Mat src;
bool stop = false;

int main(int argc, char** argv){

  float attribute[] = { 1,  1,  1,  1,  1, -1, -1, -1, -1, -1};
  int dataCount = count;
  int row, col, i, j;
  int mallocSize = sizeof(float) * dataCount * maskSize * maskSize;
  int addMem = 0;
  float rate, width = 650;
  Rect rect;

  Mat input, trainWindow;
  Mat testMat;

  trainingData = (float*) malloc(mallocSize);

  i = 1;
  input = imread(argv[i], IMREAD_GRAYSCALE);
  if(input.data == 0){
    printf("no image\n");
    return -1;
  }

  src = input.clone();
  rate = width / src.cols;
  resize(src, src, Size(width, src.rows * rate));
  while(1){
    count = dataCount;
    namedWindow("Image");
    imshow("Image", src);
    setMouseCallback("Image", onMouseTowing, (void*) &rect);
    waitKey();
    if(stop) break;
    if(addMem) trainingData = (float*) realloc(trainingData, mallocSize * addMem);

    rect.x /= rate;
    rect.y /= rate;
    rect.width /= rate;
    rect.height /= rate;
    trainWindow = input(rect).clone();
    namedWindow("Training");
    imshow("Training", trainWindow);
    setMouseCallback("Training", onMouse, (void*) &trainWindow);
    waitKey();
    addMem++;
  }
  Mat trainingDataMat(dataCount, maskSize * maskSize, CV_32FC1, trainingData);
  Mat attrMat(dataCount, 1, CV_32FC1, attribute);

  CvSVMParams params;
  params.svm_type    = CvSVM::C_SVC;
  params.kernel_type = CvSVM::LINEAR;
  params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

  CvSVM svm;
  svm.train(trainingDataMat, attrMat, Mat(), Mat(), params);

  testMat = imread(argv[argc - 1], IMREAD_GRAYSCALE);
  if(testMat.data == 0){
    printf("no image\n");
    return -1;
  }
  resize(testMat, testMat, Size(width, testMat.rows * width / testMat.cols));
  row = testMat.rows;
  col = testMat.cols;
  Mat image = Mat::zeros(row, col, CV_8U);
  float dst[row + 1][col + 1];

  float svmMax = 0;
  for (i = 0; i < row; i++){
    for (j = 0; j < col; j++){
      Mat sampleMat = sample(testMat, i, j);

      float response = svm.predict(sampleMat, true);

      if(response > 0){
        dst[i][j] = response;
        if(response > svmMax) svmMax = response;
      }
      else{
        dst[i][j] = 0;
      }

    }
  }
  for(i = 0; i < row; i++){
    for(j = 0; j < col; j++){
      image.at<uchar>(i, j) = dst[i][j] * 255.0 / svmMax;
    }
  }
  bitwise_not(image, image);

  namedWindow("Image");
  imshow("Image",image);
  waitKey();

  return 0;
}

Point min(Point p1, Point p2){
  Point point;
  point.x = p1.x < p2.x ? p1.x : p2.x;
  point.y = p1.y < p2.y ? p1.y : p2.y;
  return point;
}

void onMouseTowing(int event, int x, int y, int flags, void *param){
  static Point p1;
  static bool towing = false;
  Rect* rect = (Rect*) param;
  Mat img = src.clone();

  x = x < 0 ? 0 : x >= img.cols ? img.cols - 1 : x;
  y = y < 0 ? 0 : y >= img.rows ? img.rows - 1 : y;
  if(event == CV_EVENT_LBUTTONDOWN){
    p1.x = x;
    p1.y = y;
    towing = true;
  }
  else if(event == CV_EVENT_MOUSEMOVE && towing){
    Point p2(x, y);
    Point lt = min(p1, p2);
    *rect = Rect (lt.x, lt.y, abs(p1.x - p2.x), abs(p1.y - p2.y));
    rectangle(img, *rect, Scalar(0, 0, 255), 2);
    imshow("Image", img);
  }
  else if(event == CV_EVENT_LBUTTONUP){
    towing = false;
    cvDestroyWindow("Image");
  }
  else if(event == CV_EVENT_RBUTTONDOWN){
    stop = true;
    cvDestroyWindow("Image");
  }
}

void onMouse(int event, int x, int y, int flags, void *param){

  int i, j, k, size;
  int imgX, imgY;
  Mat* sorce = (Mat*) param;
  Mat img = (*sorce).clone();

  if(event == CV_EVENT_LBUTTONDOWN){

    circle(img, Point(x, y), 3, Scalar(0, 0, 0), -1);
    imshow("Training", img);

    size = maskSize / 2;
    for(i = -size; i <= size; i++){
      for(j = -size; j <= size; j++){
        imgX = x + i;
        imgY = y + j;
        imgX = imgX < 0 ? 0 : imgX >= img.cols ? img.cols - 1 : imgX;
        imgY = imgY < 0 ? 0 : imgY >= img.rows ? img.rows - 1 : imgY;
        trainingData[trai++] = (float)img.at<uchar>(imgX, imgY);
      }
    }

    count--;
    if(count == 0) cvDestroyWindow("Training");

  }
}

Mat sample(Mat testMat, int x, int y){

  float data[maskSize * maskSize];
  int i, j, k, size;
  int imgX, imgY;

  k = 0;

  size = maskSize / 2;
  for(i = -size; i <= size; i++){
    for(j = -size; j <= size; j++){
      imgX = x + i;
      imgY = y + j;
      imgX = imgX < 0 ? 0 : imgX >= testMat.cols ? testMat.cols - 1 : imgX;
      imgY = imgY < 0 ? 0 : imgY >= testMat.rows ? testMat.rows - 1 : imgY;
      data[k++] = (float) testMat.at<uchar>(x + i, y + j);
    }
  }
  Mat m = Mat (maskSize * maskSize, 1, CV_32FC1, data).clone();
  return m;
}
