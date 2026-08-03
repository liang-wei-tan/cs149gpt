// attention.ispc

// The 'export' keyword generates a C++ header so you can call this function
export void compute_q_k_dot_product(
    uniform const float Qi[], 
    uniform const float Kj[], 
    uniform float Sij_row[], 
    uniform const int d, 
    uniform const int Bc
) {
    // 'foreach' parallelizes this loop across the SIMD lanes.
    // If using AVX2, it processes j = 0 to 7 simultaneously.
    foreach (j = 0 ... Bc) {
        
        // This variable is private to each SIMD lane
        float total = 0.0;
        
        // 'uniform' means all SIMD lanes are on the same iteration of this loop
        for (uniform int feature = 0; feature < d; feature++) {
            
            // Qi[feature] is scalar (broadcast to all lanes)
            float qi_val = Qi[feature];
            
           // FIXED: j is now the innermost dimension. 
            // The compiler will replace the Gather with a fast contiguous Load!
            float kj_val = Kj[feature * Bc + j]; 
            
            total += (qi_val * kj_val);
        }
        
        // Write the vectorized results back to the output row
        Sij_row[j] = total;
    }
}