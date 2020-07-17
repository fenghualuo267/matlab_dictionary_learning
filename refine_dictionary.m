function [ D, alpha ] = refine_dictionary( D, X, wl1 ) % 改进字典
    
    % define the problem dimensions（尺寸）
    N  = size( X, 1 );  % length of sample vector （样本向量）
    M  = size( X, 2 );  % number of sample vectors
    Nd = size( D, 2 );  % number of dictionary atoms

    % =====================================================================
    % first step, perform（执行） sparse coding of the columns（列） (patches) of X 
    % w.r.t. the columns (atoms) of the dictionary D. Use proximal（近的） form 
    % of ADMM to perform the sparse coding task with an LLT factorization（因式）
    % of the system arising（发生） in the data term for efficiency 采用ADMM稀疏编码，和LLT因式分解提高效率
    % =====================================================================
    
    lambda     = 1.0;   % ADMM splitting（分离） penalty（处罚） weight
    gamma      = wl1;   % L1 penalty weight on sparse coding
    admm_iters = 100;    % number of ADMM iterations to perform 迭代次数
    
    % objective function for sparse coding us
    % alpha = argmin (1/2) || D alpha - X ||_2^2 + gamma || alpha ||_1
    % f(alpha) = (1/(2*gamma))|| D alpha - X ||_2^2
    % g(alpha) = || alpha ||_1
    
    % intialize the sparse coding coefficients（系数矩阵）
    alpha = randn( Nd, M );
    
    % pre-factorize（分解） the regularized（正则化） problem for efficiency
    pI = pinv( (lambda/gamma)*((D')*D) + eye( Nd ) );
    
    % defind the proximal（近的） operators（运算符） for the ADMM sparse coding operation（运算）
    prox_f = @( v ) pI*((lambda/gamma)*((D')*X) + v);
    prox_g = @( v ) max( v - lambda, 0 ) - max( -v - lambda, 0 );

    % initialize sparse coding solution, splitting variable（分离变量） and Lagrange multipliers（拉格朗日乘数）
    % multipliers
    Z = alpha;
    U = alpha-Z;

    % perform the ADMM algorithm（算法）
    for iter=1:admm_iters,
       % update the sparse coding vector
       alpha = prox_f( Z - U );

       % update the splitting variable
       Z = prox_g( alpha + U );

       % update the Lagrange multipliers
       U = U + alpha - Z;
    end
   
    
    % =====================================================================
    % second step, update the dictionary
    % =====================================================================
    % the input set of patches may be rank-deficient（满秩）, so compute the 
    % svd and solve the least squares problem using the pseudo-inverse
    %使用伪逆计算svd和解决最小的稀疏问题
    [ U, S, V ] = svd( alpha' );
    for i=1:min(size(S)),
        S(i,i) = 1.0/max( S(i,i), 1e-3);
    end
    
    % form the dictionary from the SVD and then re-normalize（正常化） the columns（列）tile
    % of D.  These are highly unlikely to be zero norm, but the paranoid could add a check
    % 
    D = ((V*(S')*(U'))*(X'))';
    for i=1:size(D,2),
       D(:,i) = D(:,i) / norm( D(:,i) ); 
    end
    
end

