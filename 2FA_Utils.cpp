// 2FA_Utils.cpp

#define NOMINMAX
#include "2FA_Utils.h" 

#include <opencv2/imgcodecs.hpp> 
#include <opencv2/imgproc.hpp>   
#include <opencv2/dnn.hpp>       
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>       
#include <filesystem>    
#include <numeric>       
#include <algorithm>     
#include <vector>
#include <cmath> // Pour std::abs

// ... (Les fonctions SaveColorFrameAsImage, LoadUser2DEmbeddings, SaveUser2DEmbeddings, 
//      calculate_cosine_similarity, PreprocessFaceForONNX, Check3DLiveness
//      restent les mêmes que dans l'artefact 2FA_Utils_cpp_08. Copiez-les ici.)

// Sauvegarde une image couleur de la Kinect, potentiellement après détection de visage.
void SaveColorFrameAsImage(
    IColorFrame* pColorFrame,
    const std::string& userOutputFolder,
    int imageIndexForUser,
    cv::CascadeClassifier& faceCascade,
    const std::string& userNameForFile
) {
    if (!pColorFrame) {
        return;
    }

    IFrameDescription* pFrameDescription = nullptr;
    int width = 0, height = 0;
    ColorImageFormat imageFormat = ColorImageFormat_None;
    UINT nColorBufferSize = 0;
    BYTE* pColorBuffer = nullptr;

    HRESULT hr = pColorFrame->get_FrameDescription(&pFrameDescription);
    if (FAILED(hr) || !pFrameDescription) {
        if (pFrameDescription) pFrameDescription->Release();
        return;
    }
    pFrameDescription->get_Width(&width);
    pFrameDescription->get_Height(&height);
    pColorFrame->get_RawColorImageFormat(&imageFormat);
    pFrameDescription->Release();

    if (width == 0 || height == 0) {
        return;
    }

    cv::Mat imageToSaveBgr;
    hr = pColorFrame->AccessRawUnderlyingBuffer(&nColorBufferSize, &pColorBuffer);
    if (FAILED(hr) || !pColorBuffer) {
        return;
    }

    cv::Mat kinectRawImage;
    if (imageFormat == ColorImageFormat_Bgra) {
        kinectRawImage = cv::Mat(height, width, CV_8UC4, pColorBuffer);
        if (!kinectRawImage.empty()) cv::cvtColor(kinectRawImage, imageToSaveBgr, cv::COLOR_BGRA2BGR);
    }
    else if (imageFormat == ColorImageFormat_Yuy2) {
        kinectRawImage = cv::Mat(height, width, CV_8UC2, pColorBuffer);
        if (!kinectRawImage.empty()) cv::cvtColor(kinectRawImage, imageToSaveBgr, cv::COLOR_YUV2BGR_YUY2);
    }
    else { // Format non géré
        return;
    }

    if (imageToSaveBgr.empty()) {
        return;
    }

    bool face_detected_in_image = false;
    if (!faceCascade.empty()) {
        cv::Mat grayImage;
        cv::cvtColor(imageToSaveBgr, grayImage, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(grayImage, grayImage);
        std::vector<cv::Rect> faces;
        faceCascade.detectMultiScale(grayImage, faces, 1.1, 4, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(80, 80));
        if (!faces.empty()) {
            face_detected_in_image = true;
        }
    }
    else {
        face_detected_in_image = true;
    }

    if (face_detected_in_image) {
        try {
            std::filesystem::path output_dir_path(userOutputFolder);
            if (!std::filesystem::exists(output_dir_path)) {
                if (!std::filesystem::create_directories(output_dir_path)) {
                    std::cerr << "SaveColorFrameAsImage: Echec critique lors de la creation du dossier: " << userOutputFolder << std::endl;
                    return;
                }
            }

            std::ostringstream filename_stream;
            filename_stream << userOutputFolder << "/" << userNameForFile << "_face_"
                << std::setw(3) << std::setfill('0') << imageIndexForUser << ".png";
            std::string final_filename = filename_stream.str();

            std::vector<int> compression_params;
            compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(3);

            if (!cv::imwrite(final_filename, imageToSaveBgr, compression_params)) {
                std::cerr << "SaveColorFrameAsImage: Echec cv::imwrite pour : " << final_filename << std::endl;
            }
        }
        catch (const cv::Exception& ex) {
            std::cerr << "SaveColorFrameAsImage: Exception OpenCV lors de la sauvegarde: " << ex.what() << std::endl;
        }
        catch (const std::filesystem::filesystem_error& fs_ex) {
            std::cerr << "SaveColorFrameAsImage: Exception Filesystem: " << fs_ex.what() << std::endl;
        }
    }
}

void LoadUser2DEmbeddings(
    const std::string& filename,
    std::map<std::string, std::vector<float>>& userEmbeddingsMap,
    size_t expectedEmbeddingSize
) {
    userEmbeddingsMap.clear();
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "INFO: Fichier d'embeddings 2D '" << filename << "' non trouve." << std::endl;
        return;
    }
    uint32_t numUsers;
    file.read(reinterpret_cast<char*>(&numUsers), sizeof(numUsers));
    if (file.fail() || file.gcount() != sizeof(numUsers)) { std::cerr << "ERREUR: Lecture numUsers depuis '" << filename << "'." << std::endl; return; }

    for (uint32_t i = 0; i < numUsers; ++i) {
        uint32_t nameLength;
        file.read(reinterpret_cast<char*>(&nameLength), sizeof(nameLength));
        if (file.fail() || file.gcount() != sizeof(nameLength)) { std::cerr << "ERREUR: Lecture nameLength (user " << i << ")." << std::endl; return; }
        if (nameLength == 0 || nameLength > 256) { std::cerr << "ERREUR: Longueur nom invalide (" << nameLength << ") user " << i << "." << std::endl; return; }
        std::string userName(nameLength, '\0');
        file.read(&userName[0], nameLength);
        if (file.fail() || file.gcount() != nameLength) { std::cerr << "ERREUR: Lecture nom (user " << i << ")." << std::endl; return; }
        uint32_t embeddingSizeInFile;
        file.read(reinterpret_cast<char*>(&embeddingSizeInFile), sizeof(embeddingSizeInFile));
        if (file.fail() || file.gcount() != sizeof(embeddingSizeInFile) || embeddingSizeInFile == 0) {
            std::cerr << "ERREUR: Lecture taille embedding pour " << userName << "." << std::endl; continue;
        }
        if (expectedEmbeddingSize > 0 && embeddingSizeInFile != expectedEmbeddingSize) {
            std::cerr << "AVERT: Taille embedding pour '" << userName << "' (" << embeddingSizeInFile
                << ") != attendu (" << expectedEmbeddingSize << "). Ignore." << std::endl;
            file.seekg(static_cast<std::streamoff>(embeddingSizeInFile) * sizeof(float), std::ios::cur);
            if (file.fail()) { std::cerr << "ERREUR: Echec saut donnees pour " << userName << "." << std::endl; return; }
            continue;
        }
        std::vector<float> embedding(embeddingSizeInFile);
        file.read(reinterpret_cast<char*>(embedding.data()), static_cast<std::streamsize>(embeddingSizeInFile) * sizeof(float));
        if (file.fail() || file.gcount() != static_cast<std::streamsize>(embeddingSizeInFile) * sizeof(float)) {
            std::cerr << "ERREUR: Lecture data embedding pour " << userName << "." << std::endl; continue;
        }
        userEmbeddingsMap[userName] = embedding;
    }
    file.close();
    std::cout << userEmbeddingsMap.size() << " embedding(s) 2D charge(s) depuis '" << filename << "'." << std::endl;
}

void SaveUser2DEmbeddings(
    const std::string& filename,
    const std::map<std::string, std::vector<float>>& userEmbeddingsMap
) {
    std::ofstream file(filename, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "ERREUR: Impossible d'ouvrir '" << filename << "' pour sauvegarde embeddings 2D." << std::endl;
        return;
    }
    uint32_t numUsers = static_cast<uint32_t>(userEmbeddingsMap.size());
    file.write(reinterpret_cast<const char*>(&numUsers), sizeof(numUsers));
    for (const auto& pair : userEmbeddingsMap) {
        const std::string& userName = pair.first;
        const std::vector<float>& embedding = pair.second;
        uint32_t nameLength = static_cast<uint32_t>(userName.length());
        if (nameLength == 0 || nameLength > 256) {
            std::cerr << "AVERT SAUVEGARDE: Nom user '" << userName << "' longueur invalide (" << nameLength << "). Ignore." << std::endl;
            continue;
        }
        file.write(reinterpret_cast<const char*>(&nameLength), sizeof(nameLength));
        file.write(userName.c_str(), nameLength);
        uint32_t embeddingSize = static_cast<uint32_t>(embedding.size());
        if (embeddingSize == 0) {
            std::cerr << "AVERT SAUVEGARDE: Embedding pour '" << userName << "' vide." << std::endl;
        }
        file.write(reinterpret_cast<const char*>(&embeddingSize), sizeof(embeddingSize));
        if (embeddingSize > 0) {
            file.write(reinterpret_cast<const char*>(embedding.data()), static_cast<std::streamsize>(embeddingSize) * sizeof(float));
        }
    }
    file.close();
    std::cout << numUsers << " embedding(s) 2D sauvegardes dans '" << filename << "'." << std::endl;
}

float calculate_cosine_similarity(const std::vector<float>& vecA, const std::vector<float>& vecB) {
    if (vecA.size() != vecB.size() || vecA.empty()) {
        return 0.0f;
    }
    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    for (size_t i = 0; i < vecA.size(); ++i) {
        dotProduct += static_cast<double>(vecA[i]) * static_cast<double>(vecB[i]);
        normA += static_cast<double>(vecA[i]) * static_cast<double>(vecA[i]);
        normB += static_cast<double>(vecB[i]) * static_cast<double>(vecB[i]);
    }
    if (normA == 0.0 || normB == 0.0) {
        return 0.0f;
    }
    double denominator = std::sqrt(normA) * std::sqrt(normB);
    if (denominator < 1e-9) {
        return 0.0f;
    }
    return static_cast<float>(dotProduct / denominator);
}

cv::Mat PreprocessFaceForONNX(const cv::Mat& faceImage, const std::vector<int64_t>& modelInputDims) {
    if (faceImage.empty()) {
        std::cerr << "PreprocessFaceForONNX: Image de visage fournie est vide." << std::endl;
        return cv::Mat();
    }
    if (modelInputDims.size() != 4) {
        std::cerr << "PreprocessFaceForONNX: modelInputDims n'a pas la taille attendue de 4." << std::endl;
        return cv::Mat();
    }

    bool isNHWC = (modelInputDims[3] == 1 || modelInputDims[3] == 3);
    bool isNCHW = (modelInputDims[1] == 1 || modelInputDims[1] == 3);
    int targetHeight, targetWidth, targetChannels;

    if (isNHWC) {
        targetHeight = static_cast<int>(modelInputDims[1]);
        targetWidth = static_cast<int>(modelInputDims[2]);
        targetChannels = static_cast<int>(modelInputDims[3]);
    }
    else if (isNCHW) {
        targetChannels = static_cast<int>(modelInputDims[1]);
        targetHeight = static_cast<int>(modelInputDims[2]);
        targetWidth = static_cast<int>(modelInputDims[3]);
    }
    else {
        std::cerr << "PreprocessFaceForONNX: Format d'entree (NHWC/NCHW) indetermine." << std::endl;
        return cv::Mat();
    }

    if (targetHeight <= 0 || targetWidth <= 0 || (targetChannels != 1 && targetChannels != 3)) {
        std::cerr << "PreprocessFaceForONNX: Dimensions cibles invalides." << std::endl;
        return cv::Mat();
    }

    cv::Mat resizedImage;
    cv::resize(faceImage, resizedImage, cv::Size(targetWidth, targetHeight), 0, 0, cv::INTER_AREA);

    cv::Mat floatImage;
    if (resizedImage.channels() == 3 && targetChannels == 3) {
        resizedImage.convertTo(floatImage, CV_32FC3);
        cv::Scalar mean_bgr(103.939, 116.779, 123.68);
        cv::subtract(floatImage, mean_bgr, floatImage);
    }
    else if (resizedImage.channels() == 1 && targetChannels == 1) {
        resizedImage.convertTo(floatImage, CV_32FC1);
        floatImage /= 255.0;
    }
    else if (resizedImage.channels() == 3 && targetChannels == 1) {
        cv::Mat gray;
        cv::cvtColor(resizedImage, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(floatImage, CV_32FC1);
        floatImage /= 255.0;
    }
    else if (resizedImage.channels() == 1 && targetChannels == 3) {
        cv::Mat bgr;
        cv::cvtColor(resizedImage, bgr, cv::COLOR_GRAY2BGR);
        bgr.convertTo(floatImage, CV_32FC3);
        cv::Scalar mean_bgr(103.939, 116.779, 123.68);
        cv::subtract(floatImage, mean_bgr, floatImage);
    }
    else {
        std::cerr << "PreprocessFaceForONNX: Incompatibilite de canaux non geree." << std::endl;
        return cv::Mat();
    }
    return floatImage;
}

// Définition de ExtractFaceROIAndCount (utilisée par GetCorrelatedFaceROI)
cv::Mat ExtractFaceROIAndCount(
    IColorFrame* pColorFrame,
    cv::CascadeClassifier& faceCascade,
    int& outTotalFacesDetected
) {
    outTotalFacesDetected = 0;
    if (!pColorFrame) {
        std::cerr << "ExtractFaceROIAndCount: pColorFrame est null." << std::endl;
        return cv::Mat();
    }
    if (faceCascade.empty()) {
        std::cerr << "ExtractFaceROIAndCount: faceCascade est vide." << std::endl;
        return cv::Mat();
    }

    IFrameDescription* pFrameDescription = nullptr;
    int width = 0, height = 0;
    ColorImageFormat imageFormat = ColorImageFormat_None;
    UINT nColorBufferSize = 0;
    BYTE* pColorBuffer = nullptr;

    HRESULT hr = pColorFrame->get_FrameDescription(&pFrameDescription);
    if (FAILED(hr) || !pFrameDescription) { if (pFrameDescription) pFrameDescription->Release(); return cv::Mat(); }
    pFrameDescription->get_Width(&width);
    pFrameDescription->get_Height(&height);
    pColorFrame->get_RawColorImageFormat(&imageFormat);
    pFrameDescription->Release();

    if (width == 0 || height == 0) { return cv::Mat(); }

    cv::Mat bgrImage;
    hr = pColorFrame->AccessRawUnderlyingBuffer(&nColorBufferSize, &pColorBuffer);
    if (FAILED(hr) || !pColorBuffer) { return cv::Mat(); }

    cv::Mat kinectRawImage;
    if (imageFormat == ColorImageFormat_Bgra) {
        kinectRawImage = cv::Mat(height, width, CV_8UC4, pColorBuffer);
        if (!kinectRawImage.empty()) cv::cvtColor(kinectRawImage, bgrImage, cv::COLOR_BGRA2BGR);
    }
    else if (imageFormat == ColorImageFormat_Yuy2) {
        kinectRawImage = cv::Mat(height, width, CV_8UC2, pColorBuffer);
        if (!kinectRawImage.empty()) cv::cvtColor(kinectRawImage, bgrImage, cv::COLOR_YUV2BGR_YUY2);
    }
    else { return cv::Mat(); }

    if (bgrImage.empty()) { return cv::Mat(); }

    cv::Mat grayImage;
    cv::cvtColor(bgrImage, grayImage, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(grayImage, grayImage);

    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(grayImage, faces, 1.1, 4, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(80, 80));

    outTotalFacesDetected = static_cast<int>(faces.size());

    if (!faces.empty()) {
        return bgrImage(faces[0]).clone();
    }
    return cv::Mat();
}


cv::Mat GetCorrelatedFaceROI(
    IColorFrame* pColorFrame,
    cv::CascadeClassifier& faceCascade,
    ICoordinateMapper* pCoordinateMapper,
    const CameraSpacePoint& livePersonHead3D,
    int& outTotal2DFacesInScene, // Ce paramètre sera mis à jour par ExtractFaceROIAndCount
    bool& outCorrelatedFaceFound
) {
    outCorrelatedFaceFound = false;
    // outTotal2DFacesInScene sera mis à jour par l'appel à ExtractFaceROIAndCount ci-dessous

    if (!pColorFrame || faceCascade.empty() || !pCoordinateMapper) {
        if (!pColorFrame) std::cerr << "GetCorrelatedFaceROI: pColorFrame est null." << std::endl;
        if (faceCascade.empty()) std::cerr << "GetCorrelatedFaceROI: faceCascade est vide." << std::endl;
        if (!pCoordinateMapper) std::cerr << "GetCorrelatedFaceROI: pCoordinateMapper est null." << std::endl;
        return cv::Mat();
    }

    // Obtenir l'image BGR et le nombre total de visages 2D
    // Note: ExtractFaceROIAndCount retourne le ROI du *premier* visage détecté par OpenCV,
    // ce qui n'est pas nécessairement celui qui correspond à livePersonHead3D.
    // Nous devons d'abord obtenir tous les rectangles de visage.

    IFrameDescription* pFrameDescription = nullptr;
    int colorWidth = 0, colorHeight = 0;
    ColorImageFormat imageFormat = ColorImageFormat_None;
    UINT nColorBufferSize = 0;
    BYTE* pColorBuffer = nullptr;

    HRESULT hr = pColorFrame->get_FrameDescription(&pFrameDescription);
    if (FAILED(hr) || !pFrameDescription) { if (pFrameDescription) pFrameDescription->Release(); return cv::Mat(); }
    pFrameDescription->get_Width(&colorWidth);
    pFrameDescription->get_Height(&colorHeight);
    pColorFrame->get_RawColorImageFormat(&imageFormat);
    pFrameDescription->Release();

    if (colorWidth == 0 || colorHeight == 0) { return cv::Mat(); }

    cv::Mat bgrImage;
    hr = pColorFrame->AccessRawUnderlyingBuffer(&nColorBufferSize, &pColorBuffer);
    if (FAILED(hr) || !pColorBuffer) { return cv::Mat(); }

    cv::Mat kinectRawImage;
    if (imageFormat == ColorImageFormat_Bgra) {
        kinectRawImage = cv::Mat(colorHeight, colorWidth, CV_8UC4, pColorBuffer);
        if (!kinectRawImage.empty()) cv::cvtColor(kinectRawImage, bgrImage, cv::COLOR_BGRA2BGR);
    }
    else if (imageFormat == ColorImageFormat_Yuy2) {
        kinectRawImage = cv::Mat(colorHeight, colorWidth, CV_8UC2, pColorBuffer);
        if (!kinectRawImage.empty()) cv::cvtColor(kinectRawImage, bgrImage, cv::COLOR_YUV2BGR_YUY2);
    }
    else { return cv::Mat(); }

    if (bgrImage.empty()) { return cv::Mat(); }

    cv::Mat grayImage;
    cv::cvtColor(bgrImage, grayImage, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(grayImage, grayImage);

    std::vector<cv::Rect> detectedFaces;
    faceCascade.detectMultiScale(grayImage, detectedFaces, 1.1, 4, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(80, 80));
    outTotal2DFacesInScene = static_cast<int>(detectedFaces.size());

    if (detectedFaces.empty()) {
        return cv::Mat();
    }

    ColorSpacePoint mappedHead2D = { 0, 0 };
    hr = pCoordinateMapper->MapCameraPointToColorSpace(livePersonHead3D, &mappedHead2D);
    if (FAILED(hr) || !std::isfinite(mappedHead2D.X) || !std::isfinite(mappedHead2D.Y)) {
        std::cerr << "GetCorrelatedFaceROI: Echec de la cartographie du point de tete 3D vers l'espace couleur 2D." << std::endl;
        return cv::Mat();
    }

    for (const auto& faceRect : detectedFaces) {
        cv::Point mappedPointCv(static_cast<int>(mappedHead2D.X), static_cast<int>(mappedHead2D.Y));
        if (faceRect.contains(mappedPointCv)) {
            outCorrelatedFaceFound = true;
            return bgrImage(faceRect).clone();
        }
    }

    // Heuristique: si un seul visage 2D est détecté au total, et qu'aucune corrélation directe n'a été faite,
    // on suppose que c'est le bon visage.
    if (outTotal2DFacesInScene == 1 && !detectedFaces.empty()) {
        outCorrelatedFaceFound = true;
        return bgrImage(detectedFaces[0]).clone();
    }

    return cv::Mat();
}

// Définition de IdentifyUser2D_AI qui utilise GetCorrelatedFaceROI
std::string IdentifyUser2D_AI(
    IColorFrame* pColorFrame,
    Ort::Session* ortSession,
    const std::map<std::string,
    std::vector<float>>&allUserReferenceEmbeddings,
    const std::vector<const char*>& inputNames,
    const std::vector<const char*>& outputNames,
    const std::vector<int64_t>& inputDims,
    size_t expectedEmbeddingSize,
    cv::CascadeClassifier& faceCascade,
    ICoordinateMapper* pCoordinateMapper,
    const CameraSpacePoint& livePersonHead3D,
    float similarityThreshold,
    float& outBestSimilarityFound,
    int& outTotal2DFacesInScene // Renommé pour correspondre à l'appelant
) {
    outBestSimilarityFound = 0.0f;
    // outTotal2DFacesInScene sera mis à jour par GetCorrelatedFaceROI
    std::string identifiedUserName = "Inconnu";

    if (!pColorFrame || !ortSession || allUserReferenceEmbeddings.empty() ||
        inputNames.empty() || outputNames.empty() || inputDims.empty() ||
        faceCascade.empty() || !pCoordinateMapper) {
        std::cerr << "IdentifyUser2D_AI: Parametres initiaux invalides." << std::endl;
        outTotal2DFacesInScene = 0; // Assurer une valeur par défaut
        return identifiedUserName;
    }
    if (inputNames[0] == nullptr || outputNames[0] == nullptr) {
        std::cerr << "IdentifyUser2D_AI: Noms d'entree/sortie ONNX non valides (nullptr)." << std::endl;
        outTotal2DFacesInScene = 0;
        return identifiedUserName;
    }

    bool correlatedFaceFound = false;
    cv::Mat correlatedFaceROI = GetCorrelatedFaceROI(
        pColorFrame,
        faceCascade,
        pCoordinateMapper,
        livePersonHead3D,
        outTotal2DFacesInScene, // Ce paramètre est mis à jour par GetCorrelatedFaceROI
        correlatedFaceFound
    );

    if (correlatedFaceFound && !correlatedFaceROI.empty()) {
        // Si un visage corrélé a été trouvé, on procède à son identification.
        identifiedUserName = IdentifyUserFromFaceROI_AI(
            correlatedFaceROI, ortSession, allUserReferenceEmbeddings,
            inputNames, outputNames, inputDims, expectedEmbeddingSize,
            similarityThreshold, outBestSimilarityFound
        );
    }
    // Si !correlatedFaceFound ou si correlatedFaceROI est vide, identifiedUserName restera "Inconnu".
    // outTotal2DFacesInScene est déjà mis à jour par GetCorrelatedFaceROI.

    return identifiedUserName;
}


// Définition de IdentifyUserFromFaceROI_AI (inchangée par rapport à la version précédente)
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
) {
    outBestSimilarityFound = 0.0f;
    std::string bestMatchUserName = "Inconnu";

    if (correlatedFaceROI.empty() || !ortSession || allUserReferenceEmbeddings.empty() || inputNames.empty() || outputNames.empty() || inputDims.empty()) {
        if (correlatedFaceROI.empty()) std::cerr << "IdentifyUserFromFaceROI_AI: correlatedFaceROI est vide." << std::endl;
        return bestMatchUserName;
    }
    if (inputNames[0] == nullptr || outputNames[0] == nullptr) {
        std::cerr << "IdentifyUserFromFaceROI_AI: Noms d'entree/sortie ONNX non valides (nullptr)." << std::endl;
        return bestMatchUserName;
    }

    cv::Mat processedInputMatHWC = PreprocessFaceForONNX(correlatedFaceROI, inputDims);
    if (processedInputMatHWC.empty()) {
        std::cerr << "IdentifyUserFromFaceROI_AI: Echec du pretraitement du ROI du visage." << std::endl;
        return "Inconnu";
    }

    std::vector<int64_t> tensorShape = inputDims;
    if (tensorShape[0] == -1) { tensorShape[0] = 1; }

    bool isNHWC = (tensorShape.size() == 4 && (tensorShape[3] == 1 || tensorShape[3] == 3));
    bool isNCHW = (tensorShape.size() == 4 && (tensorShape[1] == 1 || tensorShape[1] == 3));
    cv::Mat finalInputForTensorData;

    if (isNHWC) {
        if (processedInputMatHWC.rows != tensorShape[1] || processedInputMatHWC.cols != tensorShape[2] || processedInputMatHWC.channels() != tensorShape[3]) {
            std::cerr << "IdentifyUserFromFaceROI_AI: Incoherence de dimensions pour NHWC." << std::endl; return "Inconnu";
        }
        finalInputForTensorData = processedInputMatHWC;
    }
    else if (isNCHW) {
        if (processedInputMatHWC.rows != tensorShape[2] || processedInputMatHWC.cols != tensorShape[3] || processedInputMatHWC.channels() != tensorShape[1]) {
            std::cerr << "IdentifyUserFromFaceROI_AI: Incoherence de dimensions pour NCHW." << std::endl; return "Inconnu";
        }
        finalInputForTensorData = cv::dnn::blobFromImage(processedInputMatHWC, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F);
        if (finalInputForTensorData.dims != 4 || finalInputForTensorData.size[0] != 1 || finalInputForTensorData.size[1] != tensorShape[1] || finalInputForTensorData.size[2] != tensorShape[2] || finalInputForTensorData.size[3] != tensorShape[3]) {
            std::cerr << "IdentifyUserFromFaceROI_AI: Erreur de dimension apres transposition pour NCHW." << std::endl; return "Inconnu";
        }
    }
    else {
        std::cerr << "IdentifyUserFromFaceROI_AI: Format d'entree du modele ONNX non reconnu." << std::endl; return "Inconnu";
    }

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    size_t input_tensor_elements = 1;
    for (size_t i = 0; i < tensorShape.size(); ++i) { input_tensor_elements *= tensorShape[i]; }

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, finalInputForTensorData.ptr<float>(), input_tensor_elements,
        tensorShape.data(), tensorShape.size()
    );

    std::vector<float> currentLiveEmbedding;
    try {
        auto outputTensors = ortSession->Run(Ort::RunOptions{ nullptr }, inputNames.data(), &inputTensor, 1, outputNames.data(), 1);
        if (outputTensors.empty() || !outputTensors[0].IsTensor()) {
            std::cerr << "IdentifyUserFromFaceROI_AI: Sortie du modele invalide ou non-tenseur." << std::endl;
            return "Inconnu";
        }
        const float* outputData = outputTensors[0].GetTensorData<float>();
        auto outputShapeInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
        size_t actualEmbeddingSize = outputShapeInfo.GetElementCount();

        if (actualEmbeddingSize != expectedEmbeddingSize) {
            std::cerr << "IdentifyUserFromFaceROI_AI: Taille de l'embedding de sortie (" << actualEmbeddingSize
                << ") != taille attendue (" << expectedEmbeddingSize << ")." << std::endl;
            return "Inconnu";
        }
        currentLiveEmbedding.assign(outputData, outputData + expectedEmbeddingSize);
    }
    catch (const Ort::Exception& e) {
        std::cerr << "IdentifyUserFromFaceROI_AI: Exception ONNX Runtime lors de la generation de l'embedding live: " << e.what() << std::endl;
        return "Inconnu";
    }

    if (currentLiveEmbedding.empty()) {
        std::cerr << "IdentifyUserFromFaceROI_AI: Embedding live non genere." << std::endl;
        return "Inconnu";
    }

    float maxSimilarity = -1.0f;

    for (const auto& pair : allUserReferenceEmbeddings) {
        const std::string& enrolledUserName = pair.first;
        const std::vector<float>& referenceEmbedding = pair.second;

        if (referenceEmbedding.size() != expectedEmbeddingSize) {
            std::cerr << "IdentifyUserFromFaceROI_AI: AVERTISSEMENT - Taille d'embedding incoherente pour "
                << enrolledUserName << ". Ignore." << std::endl;
            continue;
        }
        float similarity = calculate_cosine_similarity(currentLiveEmbedding, referenceEmbedding);
        if (similarity > maxSimilarity) {
            maxSimilarity = similarity;
            bestMatchUserName = enrolledUserName;
        }
    }

    outBestSimilarityFound = maxSimilarity;

    if (maxSimilarity >= similarityThreshold) {
        return bestMatchUserName;
    }
    else {
        return "Inconnu";
    }
}


bool Check3DLiveness(
    IHighDefinitionFaceFrame* pHDFaceFrame,
    IFaceAlignment* pFaceAlignment,
    IFaceModel* pFaceModel,
    UINT32 hdVertexCount,
    std::vector<CameraSpacePoint>& tempHdFaceVerticesBuffer
) {
    if (!pHDFaceFrame || !pFaceAlignment || !pFaceModel || hdVertexCount == 0) {
        return false;
    }

    BOOLEAN bIsFaceTracked = false;
    pHDFaceFrame->get_IsFaceTracked(&bIsFaceTracked);
    if (!bIsFaceTracked) {
        return false;
    }

    FaceAlignmentQuality alignmentQuality = FaceAlignmentQuality_Low;
    HRESULT hr_align = pHDFaceFrame->GetAndRefreshFaceAlignmentResult(pFaceAlignment);
    if (SUCCEEDED(hr_align)) {
        pFaceAlignment->get_Quality(&alignmentQuality);
    }

    if (alignmentQuality != FaceAlignmentQuality_High) {
        return false;
    }

    HRESULT hr_vert = pFaceModel->CalculateVerticesForAlignment(pFaceAlignment, hdVertexCount, tempHdFaceVerticesBuffer.data());
    if (FAILED(hr_vert)) {
        return false;
    }

    const float MIN_DEPTH_DIFFERENCE_LIVENESS = 0.01f; // 1 cm 

    CameraSpacePoint noseTip = tempHdFaceVerticesBuffer[HighDetailFacePoints_NoseTip];
    CameraSpacePoint leftEye = tempHdFaceVerticesBuffer[HighDetailFacePoints_LefteyeOutercorner];
    CameraSpacePoint rightEye = tempHdFaceVerticesBuffer[HighDetailFacePoints_RighteyeOutercorner];

    if (noseTip.Z <= 0.01f || leftEye.Z <= 0.01f || rightEye.Z <= 0.01f) {
        return false;
    }

    float avgEyeZ = (leftEye.Z + rightEye.Z) / 2.0f;
    float depthDifference = std::abs(noseTip.Z - avgEyeZ);

    if (depthDifference < MIN_DEPTH_DIFFERENCE_LIVENESS) {
        return false;
    }
    return true;
}
