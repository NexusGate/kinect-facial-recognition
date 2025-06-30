// 2FA_Utils.h
#ifndef TWO_FA_UTILS_H
#define TWO_FA_UTILS_H

#include <string>
#include <vector>
#include <map>

// Inclusions OpenCV n�cessaires pour les d�clarations
#include <opencv2/objdetect.hpp> // Pour cv::CascadeClassifier
#include <opencv2/core/mat.hpp>  // Pour cv::Mat

// Inclusions Kinect n�cessaires pour les d�clarations
#include "Kinect.h"         
#include "Kinect.Face.h"    

// Inclusions ONNX Runtime
#include <onnxruntime_cxx_api.h> 


// Sauvegarde une image couleur de la Kinect.
void SaveColorFrameAsImage(
    IColorFrame* pColorFrame,
    const std::string& userOutputFolder,
    int imageIndexForUser,
    cv::CascadeClassifier& faceCascade,
    const std::string& userNameForFile
);

// Charge les embeddings 2D.
void LoadUser2DEmbeddings(
    const std::string& filename,
    std::map<std::string, std::vector<float>>& userEmbeddingsMap,
    size_t expectedEmbeddingSize
);

// Sauvegarde les embeddings 2D.
void SaveUser2DEmbeddings(
    const std::string& filename,
    const std::map<std::string, std::vector<float>>& userEmbeddingsMap
);

// Calcule la similarit� cosinus.
float calculate_cosine_similarity(
    const std::vector<float>& vecA,
    const std::vector<float>& vecB
);

// Pr�traite une image de visage pour l'entr�e du mod�le ONNX.
cv::Mat PreprocessFaceForONNX(
    const cv::Mat& faceImage,
    const std::vector<int64_t>& modelInputDims
);

// Extrait le ROI du visage 2D qui correspond � la personne 3D "vivante" et compte tous les visages 2D dans la sc�ne.
cv::Mat GetCorrelatedFaceROI(
    IColorFrame* pColorFrame,
    cv::CascadeClassifier& faceCascade,
    ICoordinateMapper* pCoordinateMapper,
    const CameraSpacePoint& livePersonHead3D,
    int& outTotal2DFacesInScene,
    bool& outCorrelatedFaceFound
);

// Identifie un utilisateur � partir d'un ROI de visage d�j� extrait.
std::string IdentifyUserFromFaceROI_AI(
    const cv::Mat& correlatedFaceROI,
    Ort::Session* ortSession,
    const std::map<std::string,
    std::vector<float>>&allUserReferenceEmbeddings,
    const std::vector<const char*>& inputNames,
    const std::vector<const char*>& outputNames,
    const std::vector<int64_t>& inputDims,
    size_t expectedEmbeddingSize,
    float similarityThreshold,
    float& outBestSimilarityFound
);

// Fonction principale d'identification 2D qui utilise GetCorrelatedFaceROI et IdentifyUserFromFaceROI_AI
std::string IdentifyUser2D_AI(
    IColorFrame* pColorFrame,
    Ort::Session* ortSession,
    const std::map<std::string, std::vector<float>>& allUserReferenceEmbeddings,
    const std::vector<const char*>& inputNames,
    const std::vector<const char*>& outputNames,
    const std::vector<int64_t>& inputDims,
    size_t expectedEmbeddingSize,
    cv::CascadeClassifier& faceCascade,
    ICoordinateMapper* pCoordinateMapper, // Ajout� pour la corr�lation
    const CameraSpacePoint& livePersonHead3D, // Ajout� pour la corr�lation
    float similarityThreshold,
    float& outBestSimilarityFound,
    int& outTotal2DFacesInScene // Param�tre de sortie pour le nombre de visages 2D
);


// V�rification de vivacit� 3D.
bool Check3DLiveness(
    IHighDefinitionFaceFrame* pHDFaceFrame,
    IFaceAlignment* pFaceAlignment,
    IFaceModel* pFaceModel,
    UINT32 hdVertexCount,
    std::vector<CameraSpacePoint>& tempHdFaceVerticesBuffer
);


#endif // TWO_FA_UTILS_H
