function [L, U] = LUdecompositionCrout(A)
    % Check if matrix is square
    [n, m] = size(A);
    if n ~= m
        error('Matrix must be square');
    end
    
    % Pre-allocate L and U
    L = zeros(n);
    U = zeros(n);
    
    % Set main diagonal of U to 1
    for i = 1:n
        U(i, i) = 1;
    end
    
    % Crout's algorithm for tridiagonal matrices
    for i = 1:n
        % Compute L(i,i) and L(i,i-1)
        if i == 1
            L(1, 1) = A(1, 1);
        else
            L(i, i-1) = A(i, i-1);  % Lower diagonal
            L(i, i) = A(i, i) - L(i, i-1) * U(i-1, i);
        end
        
        % Compute U(i,i+1)
        if i < n
            U(i, i+1) = A(i, i+1) / L(i, i);
        end
    end
end