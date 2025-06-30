#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <conio.h>
#include <cctype>
#include <map>
#include <algorithm>
#include <numeric>
#include <deque>
#include <tuple>
#include <filesystem>
#include <chrono> // Pour l'horodatage des vidéos

#define NOMINMAX
#include <windows.h>
#include <winhttp.h>

#include <DirectXMath.h>

#include "Kinect.h"
#include "Kinect.Face.h"
#include "2FA_Utils.h"

#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp> // Pour cv::VideoWriter

#pragma comment(lib, "Kinect20.lib")
#pragma comment(lib, "Kinect20.Face.lib")
#pragma comment(lib, "winhttp.lib")

using namespace DirectX;

// --- Définitions Globales et Utilitaires ---

const std::wstring TARGET_HOST = L"pc.nexusgate.space";
const INTERNET_PORT TARGET_PORT = INTERNET_DEFAULT_HTTPS_PORT;
const std::wstring TARGET_UNLOCK_ENDPOINT = L"/unlock";
const std::wstring AUTH_TOKEN = L"8c7e145d24cf79b3d9172a3bdeaf0193c92ffcb14c6b57e7a451b7a9a3f69d6e"; // Pensez à sécuriser ce token

// Constantes pour l'enregistrement vidéo
const int MAX_CONSECUTIVE_DENIED_ACCESSES = 3;
const int VIDEO_RECORDING_DURATION_S = 10;
const int VIDEO_FPS = 15; // Images par seconde pour la vidéo
const std::string VIDEO_RECORDINGS_BASE_FOLDER = "access_denied_videos";

UINT32 g_hdVertexCount = 0;
std::string g_lastEnrollStatusLine = "";
std::string g_lastSurveillanceDisplay_Info = "";
std::string g_lastSurveillanceDisplay_Decision = "";
std::string g_lastSurveillanceDisplay_Prompt = "";
std::string g_lastVideoStatusLine = "";

const SHORT CONSOLE_WIDTH = 80;
COORD g_surveillanceTitlePos = { 0, 0 };
COORD g_surveillanceBlankLine1Pos = { 0, 1 };
COORD g_surveillanceInfoPos = { 0, 2 };
COORD g_surveillanceDecisionPos = { 0, 3 };
COORD g_surveillanceBlankLine2Pos = { 0, 4 };
COORD g_surveillancePromptPos = { 0, 5 };
COORD g_requestStatusPos = { 0, 7 };
COORD g_videoStatusPos = { 0, 8 };

Ort::Env g_onnxEnv{ ORT_LOGGING_LEVEL_WARNING, "FacialRecognitionSystem" };
Ort::SessionOptions g_onnxSessionOptions;
Ort::Session* g_onnxSession = nullptr;
std::vector<std::string> g_onnxInputNamesStrings;
std::vector<std::string> g_onnxOutputNamesStrings;
std::vector<const char*> g_onnxInputNames;
std::vector<const char*> g_onnxOutputNames;
std::vector<int64_t> g_onnxInputDims;
size_t g_onnxExpectedEmbeddingSize = 0;

cv::CascadeClassifier g_faceCascade;
std::map<std::string, std::vector<float>> g_user2DEmbeddings;

const std::string HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml";
const std::string ONNX_MODEL_PATH = "face_embedding_vgg_face.onnx";
const std::string USER_2D_EMBEDDINGS_FILE = "user_2d_embeddings.bin";
const std::string USER_ENROLLMENT_IMAGES_BASE_FOLDER = "enrollment_images_2d";
const float COSINE_SIMILARITY_THRESHOLD_AI_ID = 0.60f;

const int TOTAL_2D_IMAGES_TO_COLLECT_ENROLLMENT = 300;
bool onnxInitializedSuccessfully = false;

template<class Interface>
inline void SafeRelease(Interface*& pInterfaceToRelease)
{
    if (pInterfaceToRelease != nullptr)
    {
        pInterfaceToRelease->Release();
        pInterfaceToRelease = nullptr;
    }
}

void ClearCurrentConsoleLine(COORD lineCoord) {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (GetConsoleScreenBufferInfo(hConsole, &csbi)) {
        DWORD count;
        DWORD cellCount = csbi.dwSize.X;
        FillConsoleOutputCharacter(hConsole, (TCHAR)' ', cellCount, lineCoord, &count);
        SetConsoleCursorPosition(hConsole, lineCoord);
    }
}

void ClearConsoleScreen() {
    HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE); CONSOLE_SCREEN_BUFFER_INFO csbi; DWORD count; DWORD cellCount; COORD homeCoords = { 0, 0 };
    if (hStdOut == INVALID_HANDLE_VALUE || !GetConsoleScreenBufferInfo(hStdOut, &csbi)) return;
    cellCount = csbi.dwSize.X * csbi.dwSize.Y;
    FillConsoleOutputCharacter(hStdOut, (TCHAR)' ', cellCount, homeCoords, &count);
    FillConsoleOutputAttribute(hStdOut, csbi.wAttributes, cellCount, homeCoords, &count);
    SetConsoleCursorPosition(hStdOut, homeCoords);
}

enum class AppState { INITIALIZING, SHOW_MENU, CONTINUOUS_SURVEILLANCE, ENROLL_GET_NAME, ENROLL_COLLECT_SAMPLES_GET_READY, ENROLL_COLLECT_SAMPLES_CAPTURE_POSE, ENROLL_PROCESSING_SAMPLES, EXITING };

std::vector<std::string> poseInstructions = {
    "Regardez Droit (Neutre).", "Regardez Legerement vers la Gauche.", "Regardez Legerement vers la Droite.",
    "Regardez Legerement en Haut.", "Regardez Legerement en Bas.", "Inclinez la Tete Legerement vers la Gauche.",
    "Inclinez la Tete Legerement vers la Droite."
};
int currentPoseIndex = 0;

const ULONGLONG DECISION_COOLDOWN_MS = 3000;
const ULONGLONG DECISION_MESSAGE_DISPLAY_DURATION_MS = 2500;
const ULONGLONG AUTHORIZATION_DURATION_FOR_UNLOCK_MS = 2000;
int images2DCapturedForEnrollment = 0;

struct TrackedUserSession {
    UINT64 bodyTrackingId = 0;
    bool isActive = false;
    std::string lastDecisionResult = "En attente...";
    ULONGLONG lastDecisionTimeMs = 0;
    std::string identifiedUserNameAI = "Inconnu";
    float lastAISimilarity = 0.0f;
    int last2DFaceCount = 0;
    ULONGLONG authorizationStartTimeMs = 0;
    bool unlockRequestSent = false;
    int consecutiveDeniedAccessCount = 0;
    bool isRecordingDeniedVideo = false;

    void Reset(UINT64 newTrackingId) {
        bodyTrackingId = newTrackingId;
        isActive = (newTrackingId != 0);
        lastDecisionResult = "En attente...";
        identifiedUserNameAI = "Inconnu";
        lastAISimilarity = 0.0f;
        last2DFaceCount = 0;
        authorizationStartTimeMs = 0;
        unlockRequestSent = false;
        consecutiveDeniedAccessCount = 0;
        isRecordingDeniedVideo = false;
    }
};

void SendUnlockRequestToNexusGate() {
    BOOL bResults = FALSE;
    HINTERNET hSession = NULL, hConnect = NULL, hRequest = NULL;
    std::string statusMessage;

    if (AUTH_TOKEN == L"VOTRE_TOKEN_ICI" || AUTH_TOKEN.empty()) {
        statusMessage = "Erreur: Token d'autorisation non configure dans le code.";
        HANDLE hConsole_req_err = GetStdHandle(STD_OUTPUT_HANDLE);
        SetConsoleCursorPosition(hConsole_req_err, g_requestStatusPos);
        ClearCurrentConsoleLine(g_requestStatusPos);
        std::cout << statusMessage.substr(0, CONSOLE_WIDTH - 1) << std::flush;
        return;
    }

    hSession = WinHttpOpen(L"Kinect Unlock Request Agent/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);

    if (hSession) {
        hConnect = WinHttpConnect(hSession, TARGET_HOST.c_str(), TARGET_PORT, 0);
        if (!hConnect) {
            if (statusMessage.empty()) statusMessage = "Erreur WinHttpConnect: " + std::to_string(GetLastError());
        }
    }
    else {
        statusMessage = "Erreur WinHttpOpen: " + std::to_string(GetLastError());
    }

    if (hConnect) {
        hRequest = WinHttpOpenRequest(hConnect, L"POST", TARGET_UNLOCK_ENDPOINT.c_str(), NULL, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE);
        if (!hRequest) {
            if (statusMessage.empty()) statusMessage = "Erreur WinHttpOpenRequest: " + std::to_string(GetLastError());
        }
    }

    if (hRequest) {
        std::wstring authHeader = L"Authorization: Bearer " + AUTH_TOKEN;
        bResults = WinHttpAddRequestHeaders(hRequest, authHeader.c_str(), (ULONG)-1L, WINHTTP_ADDREQ_FLAG_ADD);
        if (bResults) {
            bResults = WinHttpAddRequestHeaders(hRequest, L"Content-Type: application/json\r\n", (ULONG)-1L, WINHTTP_ADDREQ_FLAG_ADD);
        }
        if (!bResults && statusMessage.empty()) {
            statusMessage = "Erreur WinHttpAddRequestHeaders: " + std::to_string(GetLastError());
        }
    }

    if (bResults && hRequest) {
        bResults = WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0);
        if (!bResults && statusMessage.empty()) {
            statusMessage = "Erreur WinHttpSendRequest: " + std::to_string(GetLastError());
        }
    }
    else {
        if (statusMessage.empty()) statusMessage = "Erreur avant envoi (requete ou headers invalides).";
        bResults = FALSE;
    }

    if (bResults) {
        bResults = WinHttpReceiveResponse(hRequest, NULL);
        if (!bResults && statusMessage.empty()) {
            statusMessage = "Erreur WinHttpReceiveResponse: " + std::to_string(GetLastError());
        }
    }

    if (bResults) {
        DWORD dwStatusCode = 0;
        DWORD dwTempSize = sizeof(dwStatusCode);
        if (WinHttpQueryHeaders(hRequest, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, NULL, &dwStatusCode, &dwTempSize, NULL)) {
            if (dwStatusCode == 200) {
                statusMessage = "Requete Unlock HTTPS envoyee avec succes a NexusGate.";
            }
            else {
                statusMessage = "Echec requete Unlock. NexusGate a repondu: " + std::to_string(dwStatusCode);
            }
        }
        else {
            if (statusMessage.empty()) statusMessage = "Erreur WinHttpQueryHeaders: " + std::to_string(GetLastError());
        }
    }

    HANDLE hConsole_req = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleCursorPosition(hConsole_req, g_requestStatusPos);
    ClearCurrentConsoleLine(g_requestStatusPos);
    if (statusMessage.empty()) {
        statusMessage = bResults ? "Requete traitee, statut final inconnu." : "Echec requete, cause inconnue.";
    }
    std::cout << statusMessage.substr(0, CONSOLE_WIDTH - 1) << std::flush;

    if (hRequest) WinHttpCloseHandle(hRequest);
    if (hConnect) WinHttpCloseHandle(hConnect);
    if (hSession) WinHttpCloseHandle(hSession);
}

void ShowMainMenu(const std::string& lastMessage = "") {
    ClearConsoleScreen();
    std::cout << "--- MENU PRINCIPAL (SECURITE FACIALE KINECT V2 - IA IDENT) ---" << std::endl;
    std::cout << "'S': Enregistrer Nouveau Visage (Collecte images 2D pour IA)" << std::endl;
    std::cout << "'A': Activer Surveillance Continue (Vivacite 3D + ID IA 2D)" << std::endl;
    std::cout << "'L': Recharger Base de Donnees Embeddings 2D" << std::endl;
    std::cout << "'Q': Quitter" << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
    if (!lastMessage.empty()) { std::cout << "Message: " << lastMessage << std::endl; std::cout << "----------------------------------------------------------------" << std::endl; }
    std::cout << "Votre choix: " << std::flush;
}

bool InitializeONNX_Main() {
    try {
        if (!g_faceCascade.load(HAAR_CASCADE_PATH)) {
            std::cerr << "ERREUR CRITIQUE: Impossible de charger Haar Cascade depuis: " << HAAR_CASCADE_PATH << std::endl;
            return false;
        }
        std::cout << "Haar Cascade chargee avec succes." << std::endl;

        g_onnxSessionOptions.SetIntraOpNumThreads(1);
        g_onnxSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        std::wstring modelPathW(ONNX_MODEL_PATH.begin(), ONNX_MODEL_PATH.end());
        if (!std::filesystem::exists(ONNX_MODEL_PATH)) {
            std::cerr << "AVERTISSEMENT: Fichier modele ONNX '" << ONNX_MODEL_PATH << "' non trouve." << std::endl;
            onnxInitializedSuccessfully = false;
            g_onnxExpectedEmbeddingSize = 0;
            LoadUser2DEmbeddings(USER_2D_EMBEDDINGS_FILE, g_user2DEmbeddings, g_onnxExpectedEmbeddingSize);
            return true;
        }

        g_onnxSession = new Ort::Session(g_onnxEnv, modelPathW.c_str(), g_onnxSessionOptions);
        std::cout << "Modele ONNX charge avec succes depuis: " << ONNX_MODEL_PATH << std::endl;

        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = g_onnxSession->GetInputCount();
        g_onnxInputNamesStrings.assign(num_input_nodes, "");
        g_onnxInputNames.assign(num_input_nodes, nullptr);
        for (size_t i = 0; i < num_input_nodes; ++i) {
            Ort::AllocatedStringPtr name_ptr = g_onnxSession->GetInputNameAllocated(i, allocator);
            if (!name_ptr) throw std::runtime_error("Nom d'entree ONNX est null pour l'index " + std::to_string(i));
            g_onnxInputNamesStrings[i] = name_ptr.get();
        }
        for (size_t i = 0; i < num_input_nodes; ++i) g_onnxInputNames[i] = g_onnxInputNamesStrings[i].c_str();

        Ort::TypeInfo input_type_info = g_onnxSession->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        g_onnxInputDims = input_tensor_info.GetShape();

        size_t num_output_nodes = g_onnxSession->GetOutputCount();
        g_onnxOutputNamesStrings.assign(num_output_nodes, "");
        g_onnxOutputNames.assign(num_output_nodes, nullptr);
        for (size_t i = 0; i < num_output_nodes; ++i) {
            Ort::AllocatedStringPtr name_ptr = g_onnxSession->GetOutputNameAllocated(i, allocator);
            if (!name_ptr) throw std::runtime_error("Nom de sortie ONNX est null pour l'index " + std::to_string(i));
            g_onnxOutputNamesStrings[i] = name_ptr.get();
        }
        for (size_t i = 0; i < num_output_nodes; ++i) g_onnxOutputNames[i] = g_onnxOutputNamesStrings[i].c_str();

        Ort::TypeInfo output_type_info = g_onnxSession->GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_dims = output_tensor_info.GetShape();
        g_onnxExpectedEmbeddingSize = 0;

        if (!output_dims.empty()) {
            if (output_dims.size() >= 2 && (output_dims[0] == 1 || output_dims[0] == -1) && output_dims[1] > 0) {
                g_onnxExpectedEmbeddingSize = static_cast<size_t>(output_dims[1]);
            }
            else if (output_dims.size() == 1 && output_dims[0] > 0) {
                g_onnxExpectedEmbeddingSize = static_cast<size_t>(output_dims[0]);
            }
        }
        if (g_onnxExpectedEmbeddingSize == 0) {
            std::string shape_str = "["; for (size_t i = 0; i < output_dims.size(); ++i) { shape_str += std::to_string(output_dims[i]) + (i == output_dims.size() - 1 ? "" : ","); } shape_str += "]";
            throw std::runtime_error("Taille d'embedding ONNX de sortie non determinable ou invalide. Forme detectee: " + shape_str);
        }

        std::cout << "ONNX Info: Entree '" << g_onnxInputNames[0] << "', Dims: "; for (long long d : g_onnxInputDims) std::cout << d << " "; std::cout << std::endl;
        std::cout << "ONNX Info: Sortie '" << g_onnxOutputNames[0] << "', Taille Embedding Calculee: " << g_onnxExpectedEmbeddingSize << std::endl;

        LoadUser2DEmbeddings(USER_2D_EMBEDDINGS_FILE, g_user2DEmbeddings, g_onnxExpectedEmbeddingSize);
        onnxInitializedSuccessfully = true;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ERREUR CRITIQUE ONNX Runtime: " << e.what() << std::endl;
        if (g_onnxSession) { delete g_onnxSession; g_onnxSession = nullptr; }
        onnxInitializedSuccessfully = false;
        LoadUser2DEmbeddings(USER_2D_EMBEDDINGS_FILE, g_user2DEmbeddings, 0);
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "ERREUR CRITIQUE lors de l'initialisation d'ONNX: " << e.what() << std::endl;
        if (g_onnxSession) { delete g_onnxSession; g_onnxSession = nullptr; }
        onnxInitializedSuccessfully = false;
        LoadUser2DEmbeddings(USER_2D_EMBEDDINGS_FILE, g_user2DEmbeddings, 0);
        return true;
    }
    return onnxInitializedSuccessfully;
}

bool RecordDeniedAccessVideo(IColorFrameReader* pColorFrameReader, const std::string& outputFolder) {
    if (!pColorFrameReader) {
        std::cerr << "RecordDeniedAccessVideo: pColorFrameReader est null." << std::endl;
        return false;
    }

    IColorFrame* pFirstFrame = nullptr;
    IFrameDescription* pFrameDescription = nullptr;
    int frameWidth = 0, frameHeight = 0;
    ColorImageFormat imageFormat = ColorImageFormat_None;

    HRESULT hr = pColorFrameReader->AcquireLatestFrame(&pFirstFrame);
    if (SUCCEEDED(hr) && pFirstFrame) {
        hr = pFirstFrame->get_FrameDescription(&pFrameDescription);
        if (SUCCEEDED(hr) && pFrameDescription) {
            pFrameDescription->get_Width(&frameWidth);
            pFrameDescription->get_Height(&frameHeight);
            pFirstFrame->get_RawColorImageFormat(&imageFormat);
            SafeRelease(pFrameDescription);
        }
        SafeRelease(pFirstFrame);
    }

    if (frameWidth == 0 || frameHeight == 0) {
        std::cerr << "RecordDeniedAccessVideo: Impossible d'obtenir les dimensions de l'image couleur." << std::endl;
        return false;
    }

    try {
        if (!std::filesystem::exists(outputFolder)) {
            std::filesystem::create_directories(outputFolder);
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "RecordDeniedAccessVideo: Erreur creation dossier videos: " << e.what() << std::endl;
        return false;
    }

    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm buf{};
    localtime_s(&buf, &in_time_t);
    std::ostringstream videoFileNameStream;
    videoFileNameStream << outputFolder << "/denied_access_"
        << std::put_time(&buf, "%Y%m%d_%H%M%S"); // Nom de base sans extension
    std::string baseVideoFileName = videoFileNameStream.str();
    std::string finalVideoFileName; // Sera défini après le choix du codec

    cv::VideoWriter videoWriter;
    int fourcc = 0;
    bool writerOpened = false;
    std::string usedCodecName;

    // Séquence de codecs à essayer pour .mp4
    int mp4_codecs[] = {
        cv::VideoWriter::fourcc('X', '2', '6', '4'), // H.264 (x264)
        cv::VideoWriter::fourcc('H', '2', '6', '4'), // H.264
        cv::VideoWriter::fourcc('a', 'v', 'c', '1'), // H.264 (AVC)
        cv::VideoWriter::fourcc('m', 'p', '4', 'v')  // MPEG-4 Part 2
    };
    std::string mp4_codec_names[] = { "X264", "H264", "AVC1", "MP4V" };

    finalVideoFileName = baseVideoFileName + ".mp4";
    for (size_t i = 0; i < sizeof(mp4_codecs) / sizeof(mp4_codecs[0]); ++i) {
        fourcc = mp4_codecs[i];
        if (videoWriter.open(finalVideoFileName, fourcc, VIDEO_FPS, cv::Size(frameWidth, frameHeight), true)) {
            writerOpened = true;
            usedCodecName = mp4_codec_names[i];
            std::cout << "\nUtilisation du codec " << usedCodecName << " (FourCC: 0x" << std::hex << fourcc << std::dec << ") pour " << finalVideoFileName << std::endl;
            break;
        }
        else {
            std::cerr << "RecordDeniedAccessVideo: Echec ouverture avec codec " << mp4_codec_names[i] << " (0x" << std::hex << fourcc << std::dec << ") pour .mp4." << std::endl;
        }
    }

    // Si aucun codec MP4 n'a fonctionné, fallback sur MJPG avec extension .avi
    if (!writerOpened) {
        std::cerr << "RecordDeniedAccessVideo: Aucun codec MP4/H.264 n'a pu etre initialise. Tentative de fallback sur MJPG (.avi)..." << std::endl;
        finalVideoFileName = baseVideoFileName + "_fallback.avi";
        fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        if (videoWriter.open(finalVideoFileName, fourcc, VIDEO_FPS, cv::Size(frameWidth, frameHeight), true)) {
            writerOpened = true;
            usedCodecName = "MJPG";
            std::cout << "\nUtilisation du codec MJPG (FourCC: 0x" << std::hex << fourcc << std::dec << ") pour " << finalVideoFileName << std::endl;
        }
        else {
            std::cerr << "RecordDeniedAccessVideo: Echec final, impossible d'ouvrir VideoWriter meme avec MJPG pour .avi." << std::endl;
            return false;
        }
    }

    std::cout << "Enregistrement video (" << usedCodecName << ") en cours (" << VIDEO_RECORDING_DURATION_S << "s): "
        << finalVideoFileName.substr(finalVideoFileName.find_last_of("/\\") + 1) << std::endl;
    g_lastVideoStatusLine = "REC: " + finalVideoFileName.substr(finalVideoFileName.find_last_of("/\\") + 1);

    HANDLE hConsole_vid_stat = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleCursorPosition(hConsole_vid_stat, g_videoStatusPos);
    ClearCurrentConsoleLine(g_videoStatusPos);
    std::cout << g_lastVideoStatusLine.substr(0, CONSOLE_WIDTH - 1) << std::flush;

    ULONGLONG recordingStartTime = GetTickCount64();
    int framesToRecord = VIDEO_RECORDING_DURATION_S * VIDEO_FPS;
    int framesRecorded = 0;

    while (framesRecorded < framesToRecord) {
        if (GetTickCount64() - recordingStartTime > (ULONGLONG)VIDEO_RECORDING_DURATION_S * 1000 + 3000) { // Timeout de sécurité un peu plus long
            std::cerr << "RecordDeniedAccessVideo: Timeout d'enregistrement." << std::endl;
            break;
        }

        IColorFrame* pColorFrame = nullptr;
        hr = pColorFrameReader->AcquireLatestFrame(&pColorFrame);
        if (SUCCEEDED(hr) && pColorFrame) {
            UINT nColorBufferSize = 0;
            BYTE* pColorBuffer = nullptr;
            ColorImageFormat currentFrameFormat = ColorImageFormat_None;
            pColorFrame->get_RawColorImageFormat(&currentFrameFormat);

            hr = pColorFrame->AccessRawUnderlyingBuffer(&nColorBufferSize, &pColorBuffer);
            if (SUCCEEDED(hr) && pColorBuffer) {
                cv::Mat bgrImage;
                cv::Mat kinectRawImage;

                if (currentFrameFormat == ColorImageFormat_Bgra) {
                    kinectRawImage = cv::Mat(frameHeight, frameWidth, CV_8UC4, pColorBuffer);
                    if (!kinectRawImage.empty()) cv::cvtColor(kinectRawImage, bgrImage, cv::COLOR_BGRA2BGR);
                }
                else if (currentFrameFormat == ColorImageFormat_Yuy2) {
                    kinectRawImage = cv::Mat(frameHeight, frameWidth, CV_8UC2, pColorBuffer);
                    if (!kinectRawImage.empty()) cv::cvtColor(kinectRawImage, bgrImage, cv::COLOR_YUV2BGR_YUY2);
                }
                else {
                    SafeRelease(pColorFrame);
                    Sleep(1000 / VIDEO_FPS);
                    continue;
                }

                if (!bgrImage.empty()) {
                    videoWriter.write(bgrImage);
                    framesRecorded++;
                }
            }
            SafeRelease(pColorFrame);
        }
        else {
            SafeRelease(pColorFrame);
        }
        Sleep(1000 / VIDEO_FPS);
    }

    videoWriter.release();
    std::cout << "\nEnregistrement video termine: " << finalVideoFileName.substr(finalVideoFileName.find_last_of("/\\") + 1) << std::endl;
    g_lastVideoStatusLine = "Video (" + usedCodecName + ") terminee: " + finalVideoFileName.substr(finalVideoFileName.find_last_of("/\\") + 1);
    SetConsoleCursorPosition(hConsole_vid_stat, g_videoStatusPos);
    ClearCurrentConsoleLine(g_videoStatusPos);
    std::cout << g_lastVideoStatusLine.substr(0, CONSOLE_WIDTH - 1) << std::flush;

    Sleep(2000);
    g_lastVideoStatusLine.clear();
    SetConsoleCursorPosition(hConsole_vid_stat, g_videoStatusPos);
    ClearCurrentConsoleLine(g_videoStatusPos);
    return true;
}

int main()
{
    ClearConsoleScreen();
    std::cout << "Initialisation Reconnaissance Faciale (Vivacite 3D + ID IA 2D)..." << std::endl;
    IKinectSensor* pKinectSensor = nullptr; HRESULT hr;
    IBodyFrameReader* pBodyFrameReader = nullptr;
    IColorFrameReader* pColorFrameReader = nullptr;
    IHighDefinitionFaceFrameSource* pHDFaceFrameSource = nullptr;
    IHighDefinitionFaceFrameReader* pHDFaceFrameReader = nullptr;
    IFaceModel* pFaceModel = nullptr;
    IFaceAlignment* pFaceAlignment = nullptr;
    std::vector<CameraSpacePoint> hdFaceVerticesBuffer;
    ICoordinateMapper* pCoordinateMapper = nullptr;

    hr = GetDefaultKinectSensor(&pKinectSensor); if (FAILED(hr) || !pKinectSensor) { std::cerr << "ERREUR: Kinect non detecte.\n"; return -1; }
    hr = pKinectSensor->Open(); if (FAILED(hr)) { std::cerr << "ERREUR: Impossible d'ouvrir Kinect.\n"; SafeRelease(pKinectSensor); return -1; }
    std::cout << "Kinect ouvert." << std::endl;

    hr = pKinectSensor->get_CoordinateMapper(&pCoordinateMapper);
    if (FAILED(hr) || !pCoordinateMapper) { std::cerr << "ERREUR: Impossible d'obtenir CoordinateMapper.\n"; SafeRelease(pKinectSensor); return -1; }
    std::cout << "CoordinateMapper OK." << std::endl;

    IBodyFrameSource* pBodyFrameSource_ = nullptr;
    hr = pKinectSensor->get_BodyFrameSource(&pBodyFrameSource_);
    if (SUCCEEDED(hr)) { hr = pBodyFrameSource_->OpenReader(&pBodyFrameReader); }
    SafeRelease(pBodyFrameSource_);
    if (FAILED(hr) || !pBodyFrameReader) { std::cerr << "ERREUR: BodyFrameReader.\n"; SafeRelease(pCoordinateMapper); SafeRelease(pKinectSensor); return -1; }
    std::cout << "BodyFrameReader OK." << std::endl;

    IColorFrameSource* pColorFrameSource = nullptr;
    hr = pKinectSensor->get_ColorFrameSource(&pColorFrameSource);
    if (SUCCEEDED(hr)) { hr = pColorFrameSource->OpenReader(&pColorFrameReader); }
    SafeRelease(pColorFrameSource);
    if (FAILED(hr) || !pColorFrameReader) { std::cerr << "ERREUR: ColorFrameReader.\n"; SafeRelease(pBodyFrameReader); SafeRelease(pCoordinateMapper); SafeRelease(pKinectSensor); return -1; }
    std::cout << "ColorFrameReader OK." << std::endl;

    hr = CreateHighDefinitionFaceFrameSource(pKinectSensor, &pHDFaceFrameSource);
    if (SUCCEEDED(hr) && pHDFaceFrameSource) { hr = pHDFaceFrameSource->OpenReader(&pHDFaceFrameReader); }
    if (FAILED(hr) || !pHDFaceFrameSource || !pHDFaceFrameReader) { std::cerr << "ERREUR: HD Face Source/Reader.\n"; SafeRelease(pColorFrameReader); SafeRelease(pBodyFrameReader); SafeRelease(pCoordinateMapper); SafeRelease(pKinectSensor); return -1; }
    std::cout << "HD Face Source/Reader OK." << std::endl;

    hr = pHDFaceFrameSource->get_FaceModel(&pFaceModel);
    if (FAILED(hr) || !pFaceModel) { std::cerr << "ERREUR: IFaceModel HD.\n"; return -1; }
    std::cout << "IFaceModel (HD Default) OK." << std::endl;

    hr = GetFaceModelVertexCount(&g_hdVertexCount);
    if (FAILED(hr) || g_hdVertexCount == 0) { std::cerr << "ERREUR: GetFaceModelVertexCount.\n"; return -1; }
    hdFaceVerticesBuffer.resize(g_hdVertexCount);
    std::cout << "Modele HD Face avec " << g_hdVertexCount << " sommets." << std::endl;

    hr = CreateFaceAlignment(&pFaceAlignment);
    if (FAILED(hr) || !pFaceAlignment) { std::cerr << "ERREUR: IFaceAlignment HD.\n"; return -1; }
    std::cout << "IFaceAlignment (HD) OK." << std::endl;

    InitializeONNX_Main();

    if (!onnxInitializedSuccessfully && std::filesystem::exists(ONNX_MODEL_PATH)) {
        std::cerr << "--------------------------------------------------------------------" << std::endl;
        std::cerr << "AVERTISSEMENT MAJEUR: Echec de l'initialisation du systeme IA (ONNX)." << std::endl;
        std::cerr << "--------------------------------------------------------------------" << std::endl;
        Sleep(4000);
    }

    try {
        if (!std::filesystem::exists(VIDEO_RECORDINGS_BASE_FOLDER)) {
            std::filesystem::create_directories(VIDEO_RECORDINGS_BASE_FOLDER);
            std::cout << "Dossier pour videos cree: " << VIDEO_RECORDINGS_BASE_FOLDER << std::endl;
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Erreur creation dossier videos: " << e.what() << std::endl;
    }

    std::cout << "Initialisation terminee. Passage au menu principal..." << std::endl;
    Sleep(2000);

    AppState currentAppState = AppState::SHOW_MENU;
    std::string personNameToSave;
    IBody* ppBodies[BODY_COUNT] = { 0 };
    TrackedUserSession currentUserSession;
    UINT64 currentTrackedBodyIdForHD = 0;
    bool quitApp = false;
    std::string menuFeedbackMessage = "";
    bool menuNeedsRedraw = true;

    try {
        if (!std::filesystem::exists(USER_ENROLLMENT_IMAGES_BASE_FOLDER)) {
            std::filesystem::create_directories(USER_ENROLLMENT_IMAGES_BASE_FOLDER);
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Erreur creation dossier images d'enrolement: " << e.what() << std::endl;
    }

    while (!quitApp) {
        char ch_input_this_iteration = 0;
        if (_kbhit()) {
            ch_input_this_iteration = static_cast<char>(_getch());
            while (_kbhit()) { (void)_getch(); }
        }
        ULONGLONG currentTimeMs = GetTickCount64();

        switch (currentAppState) {
        case AppState::SHOW_MENU:
            if (menuNeedsRedraw) { ShowMainMenu(menuFeedbackMessage); menuFeedbackMessage = ""; menuNeedsRedraw = false; }
            if (ch_input_this_iteration != 0) {
                char choice = static_cast<char>(std::toupper(ch_input_this_iteration));
                if (choice == 'S') {
                    currentAppState = AppState::ENROLL_GET_NAME; ClearConsoleScreen();
                    std::cout << "--- ENREGISTREMENT NOUVEAU VISAGE ---" << std::endl;
                    std::cout << "Saisir le nom (sans espaces, max 30 car., 'q' pour annuler): " << std::flush;
                    personNameToSave.clear();
                    images2DCapturedForEnrollment = 0;
                }
                else if (choice == 'A') {
                    if (g_user2DEmbeddings.empty() && onnxInitializedSuccessfully) {
                        menuFeedbackMessage = "ERREUR: Base de donnees embeddings vide. Enregistrez un visage ou chargez un fichier.";
                        menuNeedsRedraw = true;
                    }
                    else if (g_user2DEmbeddings.empty() && !onnxInitializedSuccessfully && std::filesystem::exists(ONNX_MODEL_PATH)) {
                        menuFeedbackMessage = "ERREUR: ONNX non charge ET DB vide. Verifiez ONNX et enregistrez des visages.";
                        menuNeedsRedraw = true;
                    }
                    else {
                        currentAppState = AppState::CONTINUOUS_SURVEILLANCE;
                        currentUserSession.Reset(0);
                        currentTrackedBodyIdForHD = 0;
                        if (pHDFaceFrameSource) pHDFaceFrameSource->put_TrackingId(0);
                        g_lastSurveillanceDisplay_Info.clear(); g_lastSurveillanceDisplay_Decision.clear(); g_lastSurveillanceDisplay_Prompt.clear(); g_lastVideoStatusLine.clear();
                        HANDLE hConsole_clear = GetStdHandle(STD_OUTPUT_HANDLE);
                        SetConsoleCursorPosition(hConsole_clear, g_requestStatusPos); ClearCurrentConsoleLine(g_requestStatusPos);
                        SetConsoleCursorPosition(hConsole_clear, g_videoStatusPos); ClearCurrentConsoleLine(g_videoStatusPos);
                        menuNeedsRedraw = true;
                    }
                }
                else if (choice == 'L') {
                    ClearConsoleScreen(); std::cout << "Rechargement de la base de donnees embeddings 2D..." << std::endl;
                    LoadUser2DEmbeddings(USER_2D_EMBEDDINGS_FILE, g_user2DEmbeddings, g_onnxExpectedEmbeddingSize);
                    menuFeedbackMessage = "Base de donnees Embeddings 2D rechargee: " + std::to_string(g_user2DEmbeddings.size()) + " utilisateur(s).";
                    Sleep(2000); menuNeedsRedraw = true;
                }
                else if (choice == 'Q') {
                    currentAppState = AppState::EXITING;
                }
                else {
                    menuFeedbackMessage = "Choix invalide."; menuNeedsRedraw = true;
                }
            }
            Sleep(50);
            break;

        case AppState::ENROLL_GET_NAME:
            if (ch_input_this_iteration != 0) {
                char ch = ch_input_this_iteration;
                if (ch == '\r') {
                    if (!personNameToSave.empty()) {
                        currentAppState = AppState::ENROLL_COLLECT_SAMPLES_GET_READY;
                        currentPoseIndex = 0; images2DCapturedForEnrollment = 0;
                    }
                    else {
                        std::cout << "\nNom ne peut etre vide. Saisir nom: " << std::flush;
                    }
                }
                else if (ch == '\b') {
                    if (!personNameToSave.empty()) { personNameToSave.pop_back(); std::cout << "\b \b" << std::flush; }
                }
                else if (std::toupper(ch) == 'Q') {
                    currentAppState = AppState::SHOW_MENU; menuFeedbackMessage = "Enregistrement annule."; menuNeedsRedraw = true; personNameToSave.clear();
                }
                else if (personNameToSave.length() < 30 && (isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '-')) {
                    personNameToSave += ch; std::cout << ch << std::flush;
                }
            }
            else { Sleep(50); }
            break;

        case AppState::ENROLL_COLLECT_SAMPLES_GET_READY:
            ClearConsoleScreen();
            std::cout << "--- Enregistrement: " << personNameToSave << " ---" << std::endl;
            if (currentPoseIndex < poseInstructions.size()) {
                std::cout << "Pose " << (currentPoseIndex + 1) << "/" << poseInstructions.size() << ": " << poseInstructions[currentPoseIndex] << std::endl;
            }
            std::cout << "Preparez-vous... ('Q' pour annuler)" << std::endl;
            std::cout << "Collecte de " << TOTAL_2D_IMAGES_TO_COLLECT_ENROLLMENT << " images." << std::endl;
            g_lastEnrollStatusLine.clear();
            currentAppState = AppState::ENROLL_COLLECT_SAMPLES_CAPTURE_POSE;
            Sleep(2800);
            break;

        case AppState::ENROLL_COLLECT_SAMPLES_CAPTURE_POSE: {
            if (std::toupper(ch_input_this_iteration) == 'Q') {
                currentAppState = AppState::SHOW_MENU; menuFeedbackMessage = "Enregistrement interrompu."; menuNeedsRedraw = true;
                if (pHDFaceFrameSource) pHDFaceFrameSource->put_TrackingId(0); break;
            }

            std::ostringstream statusStream;
            statusStream << "Images: " << images2DCapturedForEnrollment << "/" << TOTAL_2D_IMAGES_TO_COLLECT_ENROLLMENT;
            if (currentPoseIndex < poseInstructions.size()) statusStream << " | Pose: " << poseInstructions[currentPoseIndex];
            std::string currentStatusLine = statusStream.str();

            HANDLE hConsole_enroll = GetStdHandle(STD_OUTPUT_HANDLE);
            COORD statusPos_enroll = { 0, 6 };
            if (currentStatusLine != g_lastEnrollStatusLine) {
                SetConsoleCursorPosition(hConsole_enroll, statusPos_enroll); ClearCurrentConsoleLine(statusPos_enroll); std::cout << currentStatusLine << std::flush;
                g_lastEnrollStatusLine = currentStatusLine;
            }

            IBodyFrame* pBodyFrame_enroll = nullptr; hr = pBodyFrameReader->AcquireLatestFrame(&pBodyFrame_enroll);
            if (SUCCEEDED(hr) && pBodyFrame_enroll) {
                UINT64 tempTrackIdForEnroll = 0; float minDepth = 999.0f;
                hr = pBodyFrame_enroll->GetAndRefreshBodyData(BODY_COUNT, ppBodies);
                if (SUCCEEDED(hr)) {
                    for (int i = 0; i < BODY_COUNT; ++i) if (ppBodies[i]) { BOOLEAN bT; ppBodies[i]->get_IsTracked(&bT); if (bT) { Joint j[JointType_Count]; ppBodies[i]->GetJoints(JointType_Count, j); if (j[JointType_SpineMid].Position.Z > 0 && j[JointType_SpineMid].Position.Z < minDepth) { minDepth = j[JointType_SpineMid].Position.Z; ppBodies[i]->get_TrackingId(&tempTrackIdForEnroll); } } }
                }
                for (int i = 0; i < BODY_COUNT; ++i) SafeRelease(ppBodies[i]);
                SafeRelease(pBodyFrame_enroll);

                if (tempTrackIdForEnroll != 0) {
                    IColorFrame* pColorFrame_enroll = nullptr; hr = pColorFrameReader->AcquireLatestFrame(&pColorFrame_enroll);
                    if (SUCCEEDED(hr) && pColorFrame_enroll) {
                        std::string userImageFolder = USER_ENROLLMENT_IMAGES_BASE_FOLDER + "/" + personNameToSave;
                        SaveColorFrameAsImage(pColorFrame_enroll, userImageFolder, images2DCapturedForEnrollment, g_faceCascade, personNameToSave);
                        images2DCapturedForEnrollment++;
                        SafeRelease(pColorFrame_enroll);
                        Sleep(60);
                    }
                    else { SafeRelease(pColorFrame_enroll); }
                }
                else {
                    COORD feedbackPos = { 0, 7 };
                    SetConsoleCursorPosition(hConsole_enroll, feedbackPos); ClearCurrentConsoleLine(feedbackPos);
                    std::cout << "Aucun corps detecte. Placez-vous devant la Kinect." << std::flush;
                }
            }
            else { SafeRelease(pBodyFrame_enroll); }


            if (images2DCapturedForEnrollment >= TOTAL_2D_IMAGES_TO_COLLECT_ENROLLMENT) {
                currentAppState = AppState::ENROLL_PROCESSING_SAMPLES;
            }
            else {
                int images_per_pose = TOTAL_2D_IMAGES_TO_COLLECT_ENROLLMENT / static_cast<int>(poseInstructions.size());
                if (images_per_pose == 0) images_per_pose = 1;

                if (images2DCapturedForEnrollment > 0 && (images2DCapturedForEnrollment % images_per_pose == 0)) {
                    if (currentPoseIndex < poseInstructions.size() - 1) {

                        int completedImagesForThisPoseSet = (currentPoseIndex + 1) * images_per_pose;
                        if (images2DCapturedForEnrollment >= completedImagesForThisPoseSet) {
                            currentPoseIndex++;
                            currentAppState = AppState::ENROLL_COLLECT_SAMPLES_GET_READY;
                        }
                    }
                }
            }
        } break;

        case AppState::ENROLL_PROCESSING_SAMPLES:
            ClearConsoleScreen();
            std::cout << "Collecte des " << images2DCapturedForEnrollment << " images pour " << personNameToSave << " terminee." << std::endl;
            std::cout << "IMPORTANT: Traitez ces images avec le script Python approprie pour generer/mettre a jour" << std::endl;
            std::cout << "le fichier '" << USER_2D_EMBEDDINGS_FILE << "'." << std::endl;
            std::cout << "\nAppuyez sur une touche pour retourner au menu..." << std::endl;
            (void)_getch();
            currentAppState = AppState::SHOW_MENU;
            menuFeedbackMessage = "Enrolement pour '" + personNameToSave + "' termine. Traitez les images.";
            menuNeedsRedraw = true;
            if (pHDFaceFrameSource) pHDFaceFrameSource->put_TrackingId(0);
            break;

        case AppState::CONTINUOUS_SURVEILLANCE: {
            std::string currentDecisionInfoText, identifiedUserDisplayText;

            if (ch_input_this_iteration != 0 && std::toupper(ch_input_this_iteration) == 'M') {
                currentAppState = AppState::SHOW_MENU; menuFeedbackMessage = "Surveillance arretee."; menuNeedsRedraw = true;
                if (pHDFaceFrameSource) pHDFaceFrameSource->put_TrackingId(0);
                currentUserSession.Reset(0);
                break;
            }

            IBodyFrame* pBodyFrame_surv = nullptr; hr = pBodyFrameReader->AcquireLatestFrame(&pBodyFrame_surv);
            UINT64 newTrackedBodyIdForThisFrame = 0;
            IBody* pClosestBody = nullptr;
            float minDepthForSurveillance = 999.0f;
            CameraSpacePoint currentFrameHeadPos = { 0,0,0 };

            if (SUCCEEDED(hr) && pBodyFrame_surv) {
                hr = pBodyFrame_surv->GetAndRefreshBodyData(BODY_COUNT, ppBodies);
                if (SUCCEEDED(hr)) {
                    for (int i = 0; i < BODY_COUNT; ++i) {
                        if (ppBodies[i]) {
                            BOOLEAN bTracked = false; ppBodies[i]->get_IsTracked(&bTracked);
                            if (bTracked) {
                                Joint j[JointType_Count]; ppBodies[i]->GetJoints(JointType_Count, j);
                                if (j[JointType_SpineMid].Position.Z > 0 && j[JointType_SpineMid].Position.Z < minDepthForSurveillance) {
                                    minDepthForSurveillance = j[JointType_SpineMid].Position.Z;
                                    ppBodies[i]->get_TrackingId(&newTrackedBodyIdForThisFrame);
                                    currentFrameHeadPos = j[JointType_Head].Position;
                                    if (pClosestBody) pClosestBody->Release();
                                    pClosestBody = ppBodies[i]; pClosestBody->AddRef();
                                }
                            }
                        }
                    }
                }
                for (int i = 0; i < BODY_COUNT; ++i) { if (ppBodies[i] != pClosestBody) SafeRelease(ppBodies[i]); }
            }
            SafeRelease(pBodyFrame_surv);

            if (newTrackedBodyIdForThisFrame != currentTrackedBodyIdForHD) {
                currentUserSession.Reset(newTrackedBodyIdForThisFrame);
                if (pHDFaceFrameSource) pHDFaceFrameSource->put_TrackingId(newTrackedBodyIdForThisFrame);
                currentTrackedBodyIdForHD = newTrackedBodyIdForThisFrame;
                menuNeedsRedraw = true;
            }

            bool isLivePersonThisFrame = false;
            if (currentUserSession.isActive && currentTrackedBodyIdForHD != 0 && pClosestBody) {
                IHighDefinitionFaceFrame* pHDFaceFrame_current = nullptr;
                hr = pHDFaceFrameReader->AcquireLatestFrame(&pHDFaceFrame_current);
                if (SUCCEEDED(hr) && pHDFaceFrame_current) {
                    if (g_hdVertexCount > 0 && hdFaceVerticesBuffer.size() == g_hdVertexCount) {
                        isLivePersonThisFrame = Check3DLiveness(pHDFaceFrame_current, pFaceAlignment, pFaceModel, g_hdVertexCount, hdFaceVerticesBuffer);
                    }
                    SafeRelease(pHDFaceFrame_current);
                }
            }
            SafeRelease(pClosestBody);


            if (isLivePersonThisFrame && (currentTimeMs - currentUserSession.lastDecisionTimeMs > DECISION_COOLDOWN_MS)) {
                IColorFrame* pColorFrame_AI = nullptr;
                hr = pColorFrameReader->AcquireLatestFrame(&pColorFrame_AI);
                if (SUCCEEDED(hr) && pColorFrame_AI) {
                    float currentSimilarity = 0.0f; int total2DFaces = 0;
                    std::string identifiedName = "Inconnu";
                    if (onnxInitializedSuccessfully && g_onnxSession) {
                        identifiedName = IdentifyUser2D_AI(pColorFrame_AI, g_onnxSession, g_user2DEmbeddings, g_onnxInputNames, g_onnxOutputNames, g_onnxInputDims, g_onnxExpectedEmbeddingSize, g_faceCascade, pCoordinateMapper, currentFrameHeadPos, COSINE_SIMILARITY_THRESHOLD_AI_ID, currentSimilarity, total2DFaces);
                    }
                    else {
                        total2DFaces = 0;
                    }

                    currentUserSession.last2DFaceCount = total2DFaces;
                    currentUserSession.identifiedUserNameAI = identifiedName;
                    currentUserSession.lastAISimilarity = (identifiedName != "Inconnu") ? currentSimilarity : 0.0f;
                    currentUserSession.lastDecisionTimeMs = currentTimeMs;

                    if (identifiedName != "Inconnu") {
                        currentUserSession.lastDecisionResult = "ACCES AUTORISE pour: " + identifiedName;
                        currentUserSession.consecutiveDeniedAccessCount = 0;
                        if (currentUserSession.authorizationStartTimeMs == 0) {
                            currentUserSession.authorizationStartTimeMs = currentTimeMs;
                            currentUserSession.unlockRequestSent = false;
                        }
                    }
                    else {
                        currentUserSession.lastDecisionResult = (total2DFaces > 0 || !onnxInitializedSuccessfully) ? "ACCES REFUSE: Visage non reconnu." : "ACCES REFUSE: Aucun visage detecte.";
                        if (onnxInitializedSuccessfully) {
                            currentUserSession.consecutiveDeniedAccessCount++;
                        }
                        currentUserSession.authorizationStartTimeMs = 0; currentUserSession.unlockRequestSent = false;
                    }
                    SafeRelease(pColorFrame_AI);
                }
                else { SafeRelease(pColorFrame_AI); }
            }
            else if (!isLivePersonThisFrame && currentUserSession.isActive) {
                if (currentUserSession.lastDecisionResult.find("ACCES AUTORISE") == std::string::npos || (currentTimeMs - currentUserSession.lastDecisionTimeMs > DECISION_MESSAGE_DISPLAY_DURATION_MS)) {
                    currentUserSession.lastDecisionResult = "Statut: Pas de visage 3D reel detecte.";
                }
                currentUserSession.identifiedUserNameAI = "Inconnu"; currentUserSession.lastAISimilarity = 0.0f;
                currentUserSession.authorizationStartTimeMs = 0; currentUserSession.unlockRequestSent = false;
            }

            if (isLivePersonThisFrame &&
                currentUserSession.lastDecisionResult.find("ACCES AUTORISE") != std::string::npos &&
                currentUserSession.authorizationStartTimeMs > 0 &&
                !currentUserSession.unlockRequestSent &&
                (currentTimeMs - currentUserSession.authorizationStartTimeMs >= AUTHORIZATION_DURATION_FOR_UNLOCK_MS))
            {
                SendUnlockRequestToNexusGate();
                currentUserSession.unlockRequestSent = true;
            }

            if (currentUserSession.consecutiveDeniedAccessCount >= MAX_CONSECUTIVE_DENIED_ACCESSES &&
                !currentUserSession.isRecordingDeniedVideo)
            {
                currentUserSession.isRecordingDeniedVideo = true;
                ClearCurrentConsoleLine(g_videoStatusPos);
                RecordDeniedAccessVideo(pColorFrameReader, VIDEO_RECORDINGS_BASE_FOLDER);
                currentUserSession.consecutiveDeniedAccessCount = 0;
                currentUserSession.isRecordingDeniedVideo = false;
                menuNeedsRedraw = true;
            }



            identifiedUserDisplayText = "Visage IA: " + currentUserSession.identifiedUserNameAI;
            if (currentUserSession.identifiedUserNameAI != "Inconnu") {
                std::ostringstream simStream; simStream << std::fixed << std::setprecision(3) << currentUserSession.lastAISimilarity;
                identifiedUserDisplayText += " (Sim: " + simStream.str() + ")";
            }
            identifiedUserDisplayText += " | Visages 2D: " + std::to_string(currentUserSession.last2DFaceCount);
            currentDecisionInfoText = currentUserSession.lastDecisionResult;
            if (currentUserSession.lastDecisionResult.find("ACCES AUTORISE") != std::string::npos) {
                if (currentUserSession.unlockRequestSent) {
                    currentDecisionInfoText += " (Debloque!)";
                }
                else if (currentUserSession.authorizationStartTimeMs > 0) {
                    long long timeToUnlockMs = (long long)AUTHORIZATION_DURATION_FOR_UNLOCK_MS - (currentTimeMs - currentUserSession.authorizationStartTimeMs);
                    if (timeToUnlockMs > 0) currentDecisionInfoText += " (Debloq. dans " + std::to_string(timeToUnlockMs / 1000 + 1) + "s)";
                }
            }
            else if (currentUserSession.consecutiveDeniedAccessCount > 0 && currentUserSession.consecutiveDeniedAccessCount < MAX_CONSECUTIVE_DENIED_ACCESSES) {
                currentDecisionInfoText += " (Refus: " + std::to_string(currentUserSession.consecutiveDeniedAccessCount) + "/" + std::to_string(MAX_CONSECUTIVE_DENIED_ACCESSES) + ")";
            }


            if (menuNeedsRedraw || identifiedUserDisplayText != g_lastSurveillanceDisplay_Info || currentDecisionInfoText != g_lastSurveillanceDisplay_Decision || g_lastVideoStatusLine != "") {
                HANDLE hConsole_surv = GetStdHandle(STD_OUTPUT_HANDLE);
                if (menuNeedsRedraw) { ClearConsoleScreen(); menuNeedsRedraw = false; }

                SetConsoleCursorPosition(hConsole_surv, g_surveillanceTitlePos); std::cout << "--- SURVEILLANCE (Vivacite 3D + ID IA 2D) ---" << std::string(CONSOLE_WIDTH - 46, ' ') << std::endl;
                SetConsoleCursorPosition(hConsole_surv, g_surveillanceBlankLine1Pos); std::cout << std::string(CONSOLE_WIDTH, ' ') << std::endl;

                SetConsoleCursorPosition(hConsole_surv, g_surveillanceInfoPos); std::string line1 = "Suivi ID: " + (currentTrackedBodyIdForHD == 0 ? "Aucun" : std::to_string(currentTrackedBodyIdForHD)) + " | " + identifiedUserDisplayText;
                std::cout << line1.substr(0, CONSOLE_WIDTH - 1) << std::string(std::max(0, CONSOLE_WIDTH - (int)line1.length()), ' ') << std::endl;

                SetConsoleCursorPosition(hConsole_surv, g_surveillanceDecisionPos);
                std::cout << currentDecisionInfoText.substr(0, CONSOLE_WIDTH - 1) << std::string(std::max(0, CONSOLE_WIDTH - (int)currentDecisionInfoText.length()), ' ') << std::endl;

                SetConsoleCursorPosition(hConsole_surv, g_surveillanceBlankLine2Pos); std::cout << std::string(CONSOLE_WIDTH, ' ') << std::endl;
                SetConsoleCursorPosition(hConsole_surv, g_surveillancePromptPos); std::cout << "('M': Menu Principal)" << std::string(CONSOLE_WIDTH - 22, ' ') << std::flush;

                if (!g_lastVideoStatusLine.empty()) {
                    SetConsoleCursorPosition(hConsole_surv, g_videoStatusPos);
                    ClearCurrentConsoleLine(g_videoStatusPos);
                    std::cout << g_lastVideoStatusLine.substr(0, CONSOLE_WIDTH - 1) << std::flush;
                }
                else {
                    SetConsoleCursorPosition(hConsole_surv, g_videoStatusPos); ClearCurrentConsoleLine(g_videoStatusPos);
                }


                g_lastSurveillanceDisplay_Info = identifiedUserDisplayText; g_lastSurveillanceDisplay_Decision = currentDecisionInfoText;
            }
            Sleep(50);
        } break;
        case AppState::EXITING: { quitApp = true; } break;
        default: { std::cerr << "ERREUR: Etat inconnu!" << std::endl; currentAppState = AppState::SHOW_MENU; menuNeedsRedraw = true; } break;
        }
        if (!quitApp) Sleep(10);
    }

    ClearConsoleScreen();
    std::cout << "Fermeture du systeme..." << std::endl;
    if (g_onnxSession) { delete g_onnxSession; g_onnxSession = nullptr; }
    SafeRelease(pCoordinateMapper);
    SafeRelease(pFaceAlignment); SafeRelease(pFaceModel); SafeRelease(pHDFaceFrameReader); SafeRelease(pHDFaceFrameSource);
    SafeRelease(pColorFrameReader); SafeRelease(pBodyFrameReader);
    if (pKinectSensor) { pKinectSensor->Close(); }
    SafeRelease(pKinectSensor);
    std::cout << "Programme termine." << std::endl;
    return 0;
}
