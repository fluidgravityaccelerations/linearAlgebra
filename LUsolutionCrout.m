function x = LUsolutionCrout(L, U, b)
    [n, ~] = size(L);
    
    % Forward substitution: Solve L*y = b
    y = zeros(n, 1);
    y(1) = b(1) / L(1, 1);
    for i = 2:n
        y(i) = (b(i) - L(i, i-1) * y(i-1)) / L(i, i);
    end
    
    % Backward substitution: Solve U*x = y
    x = zeros(n, 1);
    x(n) = y(n);
    for i = n-1:-1:1
        x(i) = y(i) - U(i, i+1) * x(i+1);
    end
end