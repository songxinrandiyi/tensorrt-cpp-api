#include "hungarian.h"

double HungarianAlgorithm::solve(const std::vector<std::vector<double>>& DistMatrix, std::vector<int>& Assignment) {
    int nRows = DistMatrix.size();
    int nCols = DistMatrix[0].size();

    // Flatten the distance matrix into a single vector (row-major order)
    std::vector<double> distMatrixIn(nRows * nCols);
    for (int i = 0; i < nRows; ++i)
        for (int j = 0; j < nCols; ++j)
            distMatrixIn[i * nCols + j] = DistMatrix[i][j];

    // Prepare assignment vector and cost variable
    std::vector<int> assignment(nRows, -1);
    double cost = 0.0;

    // Solve the problem
    assignmentoptimal(assignment, cost, distMatrixIn, nRows, nCols);

    // Store the result
    Assignment = std::move(assignment);
    return cost;
}

void HungarianAlgorithm::assignmentoptimal(std::vector<int>& assignment, double& cost, const std::vector<double>& distMatrixIn, int nOfRows, int nOfColumns) {
    int nOfElements = nOfRows * nOfColumns;
    int minDim = std::min(nOfRows, nOfColumns);

    cost = 0;
    assignment.assign(nOfRows, -1);

    std::vector<double> distMatrix(distMatrixIn);

    // Create state vectors
    std::vector<bool> coveredColumns(nOfColumns, false);
    std::vector<bool> coveredRows(nOfRows, false);
    std::vector<bool> starMatrix(nOfElements, false);
    std::vector<bool> primeMatrix(nOfElements, false);
    std::vector<bool> newStarMatrix(nOfElements, false);

    // Row reduction
    for (int row = 0; row < nOfRows; ++row) {
        double minValue = *std::min_element(distMatrix.begin() + row * nOfColumns, distMatrix.begin() + (row + 1) * nOfColumns);
        for (int col = 0; col < nOfColumns; ++col) {
            distMatrix[row * nOfColumns + col] -= minValue;
        }
    }

    // Column reduction
    for (int col = 0; col < nOfColumns; ++col) {
        double minValue = std::numeric_limits<double>::max();
        for (int row = 0; row < nOfRows; ++row) {
            minValue = std::min(minValue, distMatrix[row * nOfColumns + col]);
        }
        for (int row = 0; row < nOfRows; ++row) {
            distMatrix[row * nOfColumns + col] -= minValue;
        }
    }

    // Step 1: Find initial assignments
    for (int row = 0; row < nOfRows; ++row) {
        for (int col = 0; col < nOfColumns; ++col) {
            if (distMatrix[row * nOfColumns + col] == 0 && !coveredColumns[col]) {
                starMatrix[row * nOfColumns + col] = true;
                coveredColumns[col] = true;
                break;
            }
        }
    }

    // Proceed with algorithm steps
    step2a(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

    // Compute the final cost
    computeassignmentcost(assignment, cost, distMatrixIn, nOfRows, nOfColumns);
}


void HungarianAlgorithm::buildassignmentvector(std::vector<int>& assignment, const std::vector<bool>& starMatrix, int nOfRows, int nOfColumns) {
    for (int row = 0; row < nOfRows; ++row) {
        for (int col = 0; col < nOfColumns; ++col) {
            if (starMatrix[row * nOfColumns + col]) {
                assignment[row] = col;
                break;
            }
        }
    }
}

void HungarianAlgorithm::computeassignmentcost(const std::vector<int>& assignment, double& cost, const std::vector<double>& distMatrix, int nOfRows, int nOfColumns) {
    cost = 0;
    for (int row = 0; row < nOfRows; ++row) {
        int col = assignment[row];
        if (col >= 0) {
            cost += distMatrix[row * nOfColumns + col]; // Fixed indexing
        }
    }
}
void HungarianAlgorithm::step2a(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& newStarMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim) {
    for (int col = 0; col < nOfColumns; ++col) {
        for (int row = 0; row < nOfRows; ++row) {
            if (starMatrix[row * nOfColumns + col]) {
                coveredColumns[col] = true;
                break;
            }
        }
    }

    // Step 2b
    step2b(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

void HungarianAlgorithm::step2b(std::vector<int>& assignment, std::vector<double>& distMatrix, std::vector<bool>& starMatrix, std::vector<bool>& newStarMatrix, std::vector<bool>& primeMatrix, std::vector<bool>& coveredColumns, std::vector<bool>& coveredRows, int nOfRows, int nOfColumns, int minDim) {
    int coveredCount = std::count(coveredColumns.begin(), coveredColumns.end(), true);
    if (coveredCount == minDim) {
        buildassignmentvector(assignment, starMatrix, nOfRows, nOfColumns);
    } else {
        step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
    }
}

void HungarianAlgorithm::step3(
    std::vector<int>& assignment, 
    std::vector<double>& distMatrix, 
    std::vector<bool>& starMatrix, 
    std::vector<bool>& newStarMatrix, 
    std::vector<bool>& primeMatrix, 
    std::vector<bool>& coveredColumns, 
    std::vector<bool>& coveredRows, 
    int nOfRows, 
    int nOfColumns, 
    int minDim)
{
    bool zerosFound = true;
    while (zerosFound) {
        zerosFound = false;
        for (int col = 0; col < nOfColumns; ++col) {
            if (!coveredColumns[col]) {
                for (int row = 0; row < nOfRows; ++row) {
                    if (!coveredRows[row] && std::fabs(distMatrix[row * nOfColumns + col]) < std::numeric_limits<double>::epsilon()) { // Fixed indexing
                        primeMatrix[row * nOfColumns + col] = true; // Fixed indexing

                        // Look for a star in the same row
                        auto starColIt = std::find(starMatrix.begin() + row * nOfColumns, starMatrix.begin() + (row + 1) * nOfColumns, true); // Fixed range

                        if (starColIt == starMatrix.begin() + (row + 1) * nOfColumns) {
                            // No star in the same row, move to step 4
                            step4(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
                            return; // Exit early when we proceed to step 4
                        } else {
                            // There's already a star in this row, cover the row and uncover the column
                            coveredRows[row] = true;
                            coveredColumns[std::distance(starMatrix.begin() + row * nOfColumns, starColIt)] = false;
                            zerosFound = true; // Continue finding zeros
                            break;
                        }
                    }
                }
            }
        }

        // If no zero is found, proceed to step 5
        if (!zerosFound) {
            step5(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
        }
    }
}

// Step 4
void HungarianAlgorithm::step4(
    std::vector<int>& assignment, 
    std::vector<double>& distMatrix, 
    std::vector<bool>& starMatrix, 
    std::vector<bool>& newStarMatrix, 
    std::vector<bool>& primeMatrix, 
    std::vector<bool>& coveredColumns, 
    std::vector<bool>& coveredRows, 
    int nOfRows, 
    int nOfColumns, 
    int minDim, 
    int row, 
    int col)
{
    // Copy current star matrix into the new star matrix
    std::copy(starMatrix.begin(), starMatrix.end(), newStarMatrix.begin());

    // Place a star in the selected position
    newStarMatrix[row * nOfColumns + col] = true;

    // Find the column of the star in the current row and uncover it
    int starCol = std::find(starMatrix.begin() + row * nOfColumns, starMatrix.begin() + (row + 1) * nOfColumns, true) - starMatrix.begin();
    
    // Recursively go to Step 3
    step3(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

// Step 5
void HungarianAlgorithm::step5(
    std::vector<int>& assignment, 
    std::vector<double>& distMatrix, 
    std::vector<bool>& starMatrix, 
    std::vector<bool>& newStarMatrix, 
    std::vector<bool>& primeMatrix, 
    std::vector<bool>& coveredColumns, 
    std::vector<bool>& coveredRows, 
    int nOfRows, 
    int nOfColumns, 
    int minDim)
{
    std::vector<int> rowCover(nOfRows, 0);
    std::vector<int> colCover(nOfColumns, 0);

    // Copy star matrix into newStarMatrix
    std::copy(starMatrix.begin(), starMatrix.end(), newStarMatrix.begin());

    // Reset covered columns and rows
    std::fill(coveredColumns.begin(), coveredColumns.end(), false);
    std::fill(coveredRows.begin(), coveredRows.end(), false);

    // Loop through all rows and columns to find assignments
    for (int i = 0; i < nOfRows; ++i) {
        for (int j = 0; j < nOfColumns; ++j) {
            if (newStarMatrix[i * nOfColumns + j]) {
                assignment[i] = j;
                break;
            }
        }
    }

    // Final debug print: show assignments
    std::cout << "Final Assignment: ";
    for (int i = 0; i < nOfRows; ++i) {
        std::cout << assignment[i] << " ";
    }
    std::cout << std::endl;
}

