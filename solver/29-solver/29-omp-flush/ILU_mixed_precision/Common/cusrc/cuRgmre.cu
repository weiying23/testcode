

/*
    RealFlow* solve_impl(MatrixT const & matrix,
                     RealFlow* const & rhs,
                     gmres_tag const & tag,
                     PreconditionerT const & precond,
                     bool (*monitor)(RealFlow* const &, typename viennacl::result_of::cpu_value_type<typename viennacl::result_of::value_type<RealFlow*>::type>::type, void*) = NULL,
                     void *monitor_data = NULL)
  {
    typedef typename viennacl::result_of::cpu_value_type<NumericType>::type    CPU_NumericType;

    unsigned int problem_size = static_cast<unsigned int>(viennacl::traits::size(rhs));
    RealFlow* result = rhs;
    viennacl::traits::clear(result);

    vcl_size_t krylov_dim = static_cast<vcl_size_t>(tag.krylov_dim());
    if (problem_size < krylov_dim)
      krylov_dim = problem_size; 
      //A Krylov space larger than the matrix would lead to seg-faults (mathematically, error is certain to be zero already)

    RealFlow* res = rhs;
    RealFlow* v_k_tilde = rhs;
    RealFlow* v_k_tilde_temp = rhs;

    std::vector< std::vector<CPU_NumericType> > R(krylov_dim, std::vector<CPU_NumericType>(tag.krylov_dim()));
    std::vector<CPU_NumericType> projection_rhs(krylov_dim);

    std::vector<RealFlow*>          householder_reflectors(krylov_dim, rhs);
    std::vector<CPU_NumericType>  betas(krylov_dim);

    CPU_NumericType norm_rhs = viennacl::linalg::norm_2(rhs);

    if (norm_rhs <= tag.abs_tolerance()) //solution is zero if RHS norm is zero
      return result;

    tag.iters(0);

    for (unsigned int it = 0; it <= gmresmaxits; ++it)
    {
      //
      // (Re-)Initialize residual: r = b - A*x (without temporary for the result of A*x)
      //
      res = viennacl::linalg::prod(matrix, result);  //initial guess zero
      res = rhs - res;
      precond.apply(res);

      CPU_NumericType rho_0 = viennacl::linalg::norm_2(res);

      //
      // Check for premature convergence
      //
      if (rho_0 / norm_rhs < tag.tolerance() || rho_0 < tag.abs_tolerance()) // norm_rhs is known to be nonzero here
      {
        tag.error(rho_0 / norm_rhs);
        return result;
      }

      //
      // Normalize residual and set 'rho' to 1 as requested in 'A Simpler GMRES' by Walker and Zhou.
      //
      res /= rho_0;
      CPU_NumericType rho = static_cast<CPU_NumericType>(1.0);

      //
      // Iterate up until maximal Krylove space dimension is reached:
      //
      vcl_size_t k = 0;
      for (k = 0; k < krylov_dim; ++k)
      {
        tag.iters( tag.iters() + 1 ); //increase iteration counter

        // prepare storage:
        viennacl::traits::clear(R[k]);
        viennacl::traits::clear(householder_reflectors[k]);

        //compute v_k = A * v_{k-1} via Householder matrices
        if (k == 0)
        {
          v_k_tilde = viennacl::linalg::prod(matrix, res);
          precond.apply(v_k_tilde);
        }
        else
        {
          viennacl::traits::clear(v_k_tilde);
          v_k_tilde[k-1] = CPU_NumericType(1);

          //Householder rotations, part 1: Compute P_1 * P_2 * ... * P_{k-1} * e_{k-1}
          for (int i = static_cast<int>(k)-1; i > -1; --i)
            detail::gmres_householder_reflect(v_k_tilde, householder_reflectors[vcl_size_t(i)], betas[vcl_size_t(i)]);

          v_k_tilde_temp = viennacl::linalg::prod(matrix, v_k_tilde);
          precond.apply(v_k_tilde_temp);
          v_k_tilde = v_k_tilde_temp;

          //Householder rotations, part 2: Compute P_{k-1} * ... * P_{1} * v_k_tilde
          for (vcl_size_t i = 0; i < k; ++i)
            detail::gmres_householder_reflect(v_k_tilde, householder_reflectors[i], betas[i]);
        }

        //
        // Compute Householder reflection for v_k_tilde such that all entries below k-th entry are zero:
        //
        CPU_NumericType rho_k_k = 0;
        detail::gmres_setup_householder_vector(v_k_tilde, householder_reflectors[k], betas[k], rho_k_k, k);

        //
        // copy first k entries from v_k_tilde to R[k] in order to fill k-th column with result of
        // P_k * v_k_tilde = (v[0], ... , v[k-1], norm(v), 0, 0, ...) =: (rho_{1,k}, rho_{2,k}, ..., rho_{k,k}, 0, ..., 0);
        //
        detail::gmres_copy_helper(v_k_tilde, R[k], k);
        R[k][k] = rho_k_k;

        //
        // Update residual: r = P_k r
        // Set zeta_k = r[k] including machine precision considerations: mathematically we have |r[k]| <= rho
        // Set rho *= sin(acos(r[k] / rho))
        //
        detail::gmres_householder_reflect(res, householder_reflectors[k], betas[k]);

        if (res[k] > rho) //machine precision reached
          res[k] = rho;
        if (res[k] < -rho) //machine precision reached
          res[k] = -rho;
        projection_rhs[k] = res[k];

        rho *= std::sin( std::acos(projection_rhs[k] / rho) );

        printf("solve_impl-0 max_restarts:%d it:%d krylov_dim:%d k:%d left:%.5f sec:%.5f right:%.5f\n", 
          tag.max_restarts(), it, krylov_dim, k, std::fabs(rho * rho_0 / norm_rhs), std::fabs( rho_0 / norm_rhs), tag.tolerance());

        if (std::fabs(rho * rho_0 / norm_rhs) < tag.tolerance())  // Residual is sufficiently reduced, stop here
        {
          tag.error( std::fabs(rho*rho_0 / norm_rhs) );
          ++k;
          break;
        }
      } // for k

      //
      // Triangular solver stage:
      //

      for (int i2=static_cast<int>(k)-1; i2>-1; --i2)
      {
        vcl_size_t i = static_cast<vcl_size_t>(i2);
        for (vcl_size_t j=i+1; j<k; ++j)
          projection_rhs[i] -= R[j][i] * projection_rhs[j];     //R is transposed

        projection_rhs[i] /= R[i][i];
      }

      //
      // Note: 'projection_rhs' now holds the solution (eta_1, ..., eta_k)
      //

      res *= projection_rhs[0];

      if (k > 0)
      {
        for (unsigned int i = 0; i < k-1; ++i)
          res[i] += projection_rhs[i+1];
      }

      //
      // Form z inplace in 'res' by applying P_1 * ... * P_{k}
      //
      for (int i=static_cast<int>(k)-1; i>=0; --i)
        detail::gmres_householder_reflect(res, householder_reflectors[vcl_size_t(i)], betas[vcl_size_t(i)]);

      res *= rho_0;
      result += res;  // x += rho_0 * z    in the paper

      //
      // Check for convergence:
      //
      tag.error(std::fabs(rho*rho_0 / norm_rhs));

      printf("solve_impl-1 max_restarts:%d it:%d krylov_dim:%d k:%d left:%.5f error:%.5f right:%.5f\n", 
          tag.max_restarts(), it, krylov_dim, k, std::fabs(rho * rho_0 / norm_rhs),  tag.error(), tag.tolerance());

      if (monitor && monitor(result, std::fabs(rho*rho_0 / norm_rhs), monitor_data))
        break;

      if ( tag.error() < tag.tolerance() )
        return result;
    }

    return result;
  }

}
*/