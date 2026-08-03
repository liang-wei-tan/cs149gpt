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

export void compute_exp(
    uniform float Sij_row[], 
    uniform float Pij_row[], 
    uniform const int Bc
) {
    // 'foreach' parallelizes this loop across the SIMD lanes.
    // If using AVX2, it processes j = 0 to 7 simultaneously.
    foreach (j = 0 ... Bc) {
        Pij_row[j] = exp(Sij_row[j]);
    }
}

export void compute_row_sum(
    uniform float Pij[], 
    uniform float lij[],
    uniform const int Br,
    uniform const int Bc
) {
    for (uniform int i = 0; i < Br; i++) {
        float total = 0.0;
        foreach(j = 0 ... Bc){
            total += Pij[i * Bc + j];
        }
        lij[i] = reduce_add(total);
    }
}

export void compute_lnew(
    uniform float lij[],
    uniform float li[], 
    uniform float lnew[],
    uniform const int Br
) {
    foreach (i = 0 ... Br) {
        lnew[i] = lij[i] + li[i];
    }
}


export void compute_oi(
    uniform float oi[],
    uniform float li[],
    uniform float pij[],
    uniform float Vj[],
    uniform float lnew[],
    uniform const int Bc,
    uniform const int Br,
    uniform const int d
) {
    for(uniform int i = 0; i < Br; i++){
        foreach(feature = 0 ... d){
            float lo = li[i] * oi[i * d + feature];
            for(int j = 0; j < Bc; j++){
                float pval = pij[i * Bc + j];
                float vval = Vj[j * d + feature];
                lo += pval * vval;
            }
            lo = lo/lnew[i];
            oi[i * d + feature] = lo;
        }
    }
}
