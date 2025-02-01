#include <gtest/gtest.h>
#include "hungarian.h"

class HungarianAlgorithmTest : public ::testing::Test {
protected:
    HungarianAlgorithm hungarian;
};

TEST_F(HungarianAlgorithmTest, Simple2x2MatrixTest) {
    std::vector<std::vector<double>> costMatrix = {
        {1, 2},
        {2, 1}
    };

    std::vector<int> assignment;
    double cost = hungarian.solve(costMatrix, assignment);

    std::vector<int> expectedAssignment = {0, 1};

    EXPECT_EQ(assignment, expectedAssignment);
    EXPECT_DOUBLE_EQ(cost, 2.0);
}

TEST_F(HungarianAlgorithmTest, Simple3x3MatrixTest) {
    std::vector<std::vector<double>> costMatrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    std::vector<int> assignment;
    double cost = hungarian.solve(costMatrix, assignment);

    std::vector<int> expectedAssignment = {0, 1, 2};

    EXPECT_EQ(assignment, expectedAssignment);
    EXPECT_DOUBLE_EQ(cost, 15.0);
}

TEST_F(HungarianAlgorithmTest, Simple4x4MatrixTest) {
    std::vector<std::vector<double>> costMatrix = {
        {4, 2, 8, 1},
        {2, 3, 7, 6},
        {3, 1, 6, 5},
        {5, 4, 2, 7}
    };

    std::vector<int> assignment;
    double cost = hungarian.solve(costMatrix, assignment);

    std::vector<int> expectedAssignment = {3, 0, 1, 2};

    EXPECT_EQ(assignment, expectedAssignment);
    EXPECT_DOUBLE_EQ(cost, 6.0);
}

TEST_F(HungarianAlgorithmTest, Simple4x5MatrixTest) {
    std::vector<std::vector<double>> costMatrix = {
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20}
    };

    std::vector<int> assignment;
    double cost = hungarian.solve(costMatrix, assignment);

    // Expected optimal assignment (Row -> Col)
    std::vector<int> expectedAssignment = {0, 1, 2, 3};  // Assign the first 4 rows to 4 columns

    EXPECT_EQ(assignment, expectedAssignment);
    EXPECT_DOUBLE_EQ(cost, 1 + 7 + 13 + 19);  // Expected cost: 1 + 7 + 13 + 19 = 40
}

TEST_F(HungarianAlgorithmTest, Simple5x4MatrixTest) {
    std::vector<std::vector<double>> costMatrix = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16},
        {17, 18, 19, 20}
    };

    std::vector<int> assignment;
    double cost = hungarian.solve(costMatrix, assignment);

    // Expected optimal assignment (Row -> Col)
    std::vector<int> expectedAssignment = {0, 1, 2, 3};  // Assign the first 4 rows to 4 columns

    EXPECT_EQ(assignment, expectedAssignment);
    EXPECT_DOUBLE_EQ(cost, 1 + 6 + 11 + 16);  // Expected cost: 1 + 6 + 11 + 16 = 34
}

TEST_F(HungarianAlgorithmTest, Simple5x5MatrixTest) {
    std::vector<std::vector<double>> costMatrix = {
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25}
    };

    std::vector<int> assignment;
    double cost = hungarian.solve(costMatrix, assignment);

    // Expected optimal assignment (Row -> Col)
    std::vector<int> expectedAssignment = {0, 1, 2, 3, 4};  // Assign the rows to the columns in order

    EXPECT_EQ(assignment, expectedAssignment);
    EXPECT_DOUBLE_EQ(cost, 1 + 7 + 13 + 19 + 25);  // Expected cost: 1 + 7 + 13 + 19 + 25 = 65
}
