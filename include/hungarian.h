#pragma once

#include <iostream>
#include <vector>
#include <memory>  
#include <algorithm>  
#include <cmath>  
#include <limits>
#include <iterator>
#include <cfloat>

class HungarianAlgorithm
{
public:
    HungarianAlgorithm() = default;
    ~HungarianAlgorithm() = default;

    double solve(const std::vector<std::vector<double>>& DistMatrix, std::vector<int>& Assignment);

private:
    void assignmentoptimal(std::vector<int>& assignment, double& cost, const std::vector<double>& distMatrixIn, int nOfRows, int nOfColumns);
    void buildassignmentvector(std::vector<int>& assignment, const std::vector<bool>& starMatrix, int nOfRows, int nOfColumns);
    void computeassignmentcost(const std::vector<int>& assignment, double& cost, const std::vector<double>& distMatrix, int nOfRows, int nOfColumns);
    void step2a(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& newStarMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim);
    void step2b(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& newStarMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim);
    void step3(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& newStarMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim);
    void step4(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& newStarMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim, int row, int col);
    void step5(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& newStarMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim);
};



