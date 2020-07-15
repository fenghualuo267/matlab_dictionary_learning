function [ D, alpha, A, B ] = online_dictionary_learning( t, D, A, B, X, wl1 )
    
    % define the problem dimensions（尺寸）
    N  = size( X, 1 );  % length of sample vector（向量） 
    M  = size( X, 2 );  % number of sample vectors
    Nd = size( D, 2 );  % number of dictionary atoms

    % =====================================================================
    % first step, perform sparse coding of the columns (patches) of X      对X的列(patch)进行稀疏编码
    % w.r.t. the columns (atoms) of the dictionary D. Use proximal form 
    % of ADMM to perform the sparse coding task with an LLT factorization
    % of the system arising in the data term for efficiency   采用ADMM稀疏编码，和LLT因式分解提高效率
    % =====================================================================
    
    lambda     = 1.0;   % ADMM splitting penalty weight admm分解处罚权重（作用是啥）
    gamma      = wl1;   % L1 penalty（处罚） weight on sparse coding
    admm_iters = 200;   % number of ADMM iterations to perform（执行）
    
    % objective function for sparse coding us
    % alpha = argmin (1/2) || D alpha - X ||_2^2 + gamma || alpha ||_1
    % f(alpha) = (1/(2*gamma))|| D alpha - X ||_2^2
    % g(alpha) = || alpha ||_1
    
    % intialize the sparse coding coefficients 初始化稀疏编码系数
    alpha = randn( Nd, M );
    
    
    % pre-factorize the regularized problem for efficiency        前向分解的正则化为了提高效率
    pI = pinv( (lambda/gamma)*((D')*D) + eye( Nd ) );
    
    % defind the proximal（最近） operators（运算符） for the ADMM sparse coding operation（操作）
    prox_f = @( v ) pI*((lambda/gamma)*((D')*X) + v);
    prox_g = @( v ) max( v - lambda, 0 ) - max( -v - lambda, 0 );

    % initialize sparse coding solution, splitting（分割） variable（变量） and Lagrang multipliers(拉格朗日乘数）
    Z = alpha;
    U = alpha-Z;

    % perform  ADMM algorithm（算法） perform执行
    for iter=1:admm_iters,
       % update the sparse coding vector（向量）
       alpha = prox_f( Z - U );

       % update the splitting（分割） variable
       Z = prox_g( alpha + U );

       % update the Lagrange multipliers
       U = U + alpha - Z;
    end
    
    % =====================================================================
    % second step, update the dictionary
    % =====================================================================
    % update A and B（a，b是啥）
    if t < size(X,2),
        theta = t*size(X,2);
    else
        theta = size(X,2)^2+t-size(X,2);
    end
    beta  = (theta+1-size(X,2))/(theta+1);
    A = beta*A + alpha*alpha';
    B = beta*B + X*alpha';
    
    % update each column of the dictionary sequentially 顺序更新字典的每一列
    for j=1:size( D, 2 ),
        u = (B(:,j) - D*alpha(:,j))/A(j,j) + D(:,j);
        D(:,j) = u/max( norm(u), 1 );
    end
    

end
