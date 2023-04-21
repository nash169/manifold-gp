// Parameters
lc = DefineNumber[ 0.005, Name "Parameters/lc" ];
h = DefineNumber[ 0.05, Name "Parameters/h" ];
order = DefineNumber[ 3, Name "Parameters/order" ];


Point(1) = {-1, 0.5, 0, lc};
Point(2) = {-1.5, -0, 0, lc};
Point(3) = {-1, -0.5, 0, lc};
Point(4) = {-1, 0, 0, lc};

Point(5) = {1, 0.5, 0, lc};
Point(6) = {1.5, -0, 0, lc};
Point(7) = {1, -0.5, 0, lc};
Point(8) = {1, 0, 0, lc};

Point(13) = {-0.8, 0.5, -0, lc};
Point(14) = {-0.6, 0.3, -0, lc};

Point(15) = {-0.4, h, -0, lc};
Point(10) = {-0.1, h, 0, lc};
Point(12) = {0.1, h, 0, lc};
Point(18) = {0.4, h, -0, lc};

Point(16) = {0.8, 0.5, -0, 1};
Point(17) = {0.6, 0.3, -0, 1};

Point(19) = {-0.8, -0.5, -0, lc};
Point(20) = {-0.6, -0.3, -0, lc};

Point(9) = {-0.1, -h, 0, lc};
Point(21) = {-0.4, -h, -0, lc};
Point(11) = {0.1, -h, 0, lc};
Point(24) = {0.4, -h, -0, lc};

Point(22) = {0.8, -0.5, -0, lc};
Point(23) = {0.6, -0.3, -0, lc};

Circle(1) = {1, 4, 2};
Circle(2) = {2, 4, 3};

Circle(3) = {5, 8, 6};
Circle(4) = {6, 8, 7};

BSpline(5) = {1, 13, 14, 15, 10, 12, 18, 17, 16, 5};
BSpline(6) = {3, 19, 20, 21, 9, 11, 24, 23, 22, 7};

Physical Curve(7) = {1, 2, 6, 4, 3, 5};

// // Generate 1D mesh
// Mesh 1;
// SetOrder order;
// Mesh.MshFileVersion = 2.2;
//+

