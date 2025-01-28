#pragma once

#include <vector>
#include <string>

struct Config {
    static inline std::vector<std::string> classNames = {
        "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
        "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
        "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
        "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
        "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
        "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
        "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
        "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
        "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
        "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
        "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
        "teddy bear",     "hair drier", "toothbrush"
    };

    static inline std::map<int, std::string> classMapping = {
        {0, "person"},         {1, "bicycle"},          {2, "car"},           {3, "motorcycle"},    {4, "airplane"},    {5, "bus"},
        {6, "train"},          {7, "truck"},            {8, "boat"},          {9, "traffic light"}, {10, "fire hydrant"},
        {11, "stop sign"},     {12, "parking meter"},   {13, "bench"},        {14, "bird"},         {15, "cat"},       
        {16, "dog"},           {17, "horse"},           {18, "sheep"},        {19, "cow"},          {20, "elephant"},
        {21, "bear"},          {22, "zebra"},           {23, "giraffe"},      {24, "backpack"},     {25, "umbrella"},   
        {26, "handbag"},       {27, "tie"},             {28, "suitcase"},     {29, "frisbee"},      {30, "skis"},
        {31, "snowboard"},     {32, "sports ball"},     {33, "kite"},         {34, "baseball bat"}, {35, "baseball glove"},
        {36, "skateboard"},    {37, "surfboard"},       {38, "tennis racket"},{39, "bottle"},       {40, "wine glass"},
        {41, "cup"},           {42, "fork"},            {43, "knife"},        {44, "spoon"},        {45, "bowl"},
        {46, "banana"},        {47, "apple"},           {48, "sandwich"},     {49, "orange"},       {50, "broccoli"},
        {51, "carrot"},        {52, "hot dog"},         {53, "pizza"},        {54, "donut"},        {55, "cake"},
        {56, "chair"},         {57, "couch"},           {58, "potted plant"}, {59, "bed"},          {60, "dining table"},
        {61, "toilet"},        {62, "tv"},              {63, "laptop"},       {64, "mouse"},        {65, "remote"},
        {66, "keyboard"},      {67, "cell phone"},      {68, "microwave"},    {69, "oven"},         {70, "toaster"},
        {71, "sink"},          {72, "refrigerator"},    {73, "book"},         {74, "clock"},        {75, "vase"},
        {76, "scissors"},      {77, "teddy bear"},      {78, "hair drier"},   {79, "toothbrush"}
    };

    static inline const std::vector<std::vector<float>> COLOR_LIST = {
        {1, 1, 1},
        {0.098, 0.325, 0.850},
        {0.125, 0.694, 0.929},
        {0.556, 0.184, 0.494},
        {0.188, 0.674, 0.466},
        {0.933, 0.745, 0.301},
        {0.184, 0.078, 0.635},
        {0.300, 0.300, 0.300},
        {0.600, 0.600, 0.600},
        {0.000, 0.000, 1.000},
        {0.000, 0.500, 1.000},
        {0.000, 0.749, 0.749},
        {0.000, 1.000, 0.000},
        {1.000, 0.000, 0.000},
        {1.000, 0.000, 0.667},
        {0.000, 0.333, 0.333},
        {0.000, 0.667, 0.333},
        {0.000, 1.000, 0.333},
        {0.000, 0.333, 0.667},
        {0.000, 0.667, 0.667},
        {0.000, 1.000, 0.667},
        {0.000, 0.333, 1.000},
        {0.000, 0.667, 1.000},
        {0.000, 1.000, 1.000},
        {0.500, 0.333, 0.000},
        {0.500, 0.667, 0.000},
        {0.500, 1.000, 0.000},
        {0.500, 0.000, 0.333},
        {0.500, 0.333, 0.333},
        {0.500, 0.667, 0.333},
        {0.500, 1.000, 0.333},
        {0.500, 0.000, 0.667},
        {0.500, 0.333, 0.667},
        {0.500, 0.667, 0.667},
        {0.500, 1.000, 0.667},
        {0.500, 0.000, 1.000},
        {0.500, 0.333, 1.000},
        {0.500, 0.667, 1.000},
        {0.500, 1.000, 1.000},
        {1.000, 0.333, 0.000},
        {1.000, 0.667, 0.000},
        {1.000, 1.000, 0.000},
        {1.000, 0.000, 0.333},
        {1.000, 0.333, 0.333},
        {1.000, 0.667, 0.333},
        {1.000, 1.000, 0.333},
        {1.000, 0.000, 0.667},
        {1.000, 0.333, 0.667},
        {1.000, 0.667, 0.667},
        {1.000, 1.000, 0.667},
        {1.000, 0.000, 1.000},
        {1.000, 0.333, 1.000},
        {1.000, 0.667, 1.000},
        {0.000, 0.000, 0.333},
        {0.000, 0.000, 0.500},
        {0.000, 0.000, 0.667},
        {0.000, 0.000, 0.833},
        {0.000, 0.000, 1.000},
        {0.000, 0.167, 0.000},
        {0.000, 0.333, 0.000},
        {0.000, 0.500, 0.000},
        {0.000, 0.667, 0.000},
        {0.000, 0.833, 0.000},
        {0.000, 1.000, 0.000},
        {0.167, 0.000, 0.000},
        {0.333, 0.000, 0.000},
        {0.500, 0.000, 0.000},
        {0.667, 0.000, 0.000},
        {0.833, 0.000, 0.000},
        {1.000, 0.000, 0.000},
        {0.000, 0.000, 0.000},
        {0.143, 0.143, 0.143},
        {0.286, 0.286, 0.286},
        {0.429, 0.429, 0.429},
        {0.571, 0.571, 0.571},
        {0.714, 0.714, 0.714},
        {0.857, 0.857, 0.857},
        {0.741, 0.447, 0.000},
        {0.741, 0.717, 0.314},
        {0.000, 0.500, 0.500}
    };

    static inline const std::vector<std::vector<unsigned int>> KPS_COLORS = {
        {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},   {255, 128, 0},
        {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0}, {51, 153, 255},
        {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}
    };

    static inline const std::vector<std::vector<unsigned int>> SKELETON = {
        {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13},
        {6, 7},   {6, 8},   {7, 9},   {8, 10},  {9, 11},  {2, 3},  {1, 2},
        {1, 3},   {2, 4},   {3, 5},   {4, 6},   {5, 7}
    };

    static inline const std::vector<std::vector<unsigned int>> LIMB_COLORS = {
        {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {51, 153, 255}, {255, 51, 255}, {255, 51, 255}, {255, 51, 255},
        {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {255, 128, 0},  {0, 255, 0},    {0, 255, 0},
        {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0},    {0, 255, 0}
    };
};

