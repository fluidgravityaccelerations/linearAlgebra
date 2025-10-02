clear all
close all
clc

% Test matrix
A = [2, 1, 0, 0;
     1, 2, 1, 0;
     0, 1, 2, 1;
     0, 0, 1, 2];
b = [1; 2; 3; 4];

[L, U] = LUdecompositionCrout(A);
x = LUsolutionCrout(L, U, b);

% Check result
error = norm(A*x - b);